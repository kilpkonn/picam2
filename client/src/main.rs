use std::net::TcpListener;

use image::GenericImage;
use image::{ImageBuffer, Rgb};
use ndarray::{self, Array3, ArrayViewD, Axis};
use ort::{execution_providers::CUDAExecutionProvider, session::Session, value::Value};
use rust_faces::{
    priorboxes::{PriorBoxes, PriorBoxesParams},
    Face, Nms, Rect,
};
use std::io::Write;
use zenoh::{Config, Wait};

fn main() {
    tracing_subscriber::fmt::init();
    let session = zenoh::open(Config::default()).wait().unwrap();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
        .commit()
        .unwrap();

    let model = Session::builder()
        .unwrap()
        .commit_from_url(
            "https://github.com/rustybuilder/model-zoo/raw/main/face-detection/blazeface-320.onnx",
        )
        .unwrap();

    let subscriber = session.declare_subscriber("video").wait().unwrap();

    let listener = TcpListener::bind("0.0.0.0:8080").unwrap();

    loop {
        let (mut stream, _) = listener.accept().expect("Failed to accept connection");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        );

        stream.write_all(response.as_bytes()).unwrap();

        while let Ok(sample) = subscriber.recv() {
            let payload: Vec<u8> = sample.payload().to_bytes().to_vec();
            // let image = Mat::new_rows_cols_with_data(1248, 736, &payload).unwrap();
            let image = Array3::from_shape_vec([736, 1248, 3], payload).unwrap();
            let faces = detect(&model, image.view().into_dyn());

            let mut image = to_rbg8(&image);
            draw_faces(&mut image, faces);

            let encoded = turbojpeg::compress_image(&image, 95, turbojpeg::Subsamp::None).unwrap();

            dbg!(&encoded);

            let image_data = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                encoded.len()
            );

            stream.write_all(image_data.as_bytes()).unwrap();
            stream.write_all(&encoded).unwrap();
            stream.write_all(b"\r\n").unwrap();
            stream.flush().unwrap();
        }
    }
}

fn to_rbg8(arr: &Array3<u8>) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let (height, width, _) = arr.dim();
    let mut image = ImageBuffer::new(width as u32, height as u32);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let r = arr[[y as usize, x as usize, 0]];
        let g = arr[[y as usize, x as usize, 1]];
        let b = arr[[y as usize, x as usize, 2]];
        *pixel = image::Rgb([b, g, r]);
    }
    image
}

// #[derive(Debug, Clone, Copy)]
// pub struct Rect {
//     pub x: f32,
//     pub y: f32,
//     pub width: f32,
//     pub height: f32,
// }

// #[derive(Debug, Clone)]
// pub struct Face {
//     pub rect: Rect,
//     pub confidence: f32,
//     pub landmarks: Option<Vec<(f32, f32)>>,
// }

fn detect(session: &Session, image: ArrayViewD<u8>) -> Vec<Face> {
    let shape = image.shape().to_vec();
    let (width, height, _) = (shape[1], shape[0], shape[2]);

    let image = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(
        width as u32,
        height as u32,
        image.as_slice().unwrap(),
    )
    .unwrap();

    let (input_width, input_height) = image.dimensions();
    let image = Array3::<f32>::from_shape_fn(
        (3, input_height as usize, input_width as usize),
        |(c, y, x)| {
            match c {
                // https://github.com/zineos/blazeface/blob/main/tools/test.py seems to use OpenCV's BGR
                0 => image.get_pixel(x as u32, y as u32)[2] as f32 - 104.0,
                1 => image.get_pixel(x as u32, y as u32)[1] as f32 - 117.0,
                2 => image.get_pixel(x as u32, y as u32)[0] as f32 - 123.0,
                _ => unreachable!(),
            }
        },
    )
    .insert_axis(Axis(0));

    let arr = image.as_standard_layout().into_dyn();
    let arr = Value::from_array(&arr).unwrap();
    let output_tensors = session.run(ort::inputs![arr].unwrap()).unwrap();

    // Boxes regressions: N box with the format [start x, start y, end x, end y].
    let boxes: ArrayViewD<f32> = output_tensors[0].try_extract_tensor().unwrap();
    let scores: ArrayViewD<f32> = output_tensors[1].try_extract_tensor().unwrap();
    let landmarks: ArrayViewD<f32> = output_tensors[2].try_extract_tensor().unwrap();
    let num_boxes = boxes.view().shape()[1];

    let priors = PriorBoxes::new(
        &PriorBoxesParams::default(),
        (input_width as usize, input_height as usize),
    );

    let ratio = 1.0;
    let scale_ratios = (input_width as f32 / ratio, input_height as f32 / ratio);

    let faces = boxes
        .view()
        .to_shape((num_boxes, 4))
        .unwrap()
        .axis_iter(Axis(0))
        .zip(
            landmarks
                .view()
                .to_shape((num_boxes, 10))
                .unwrap()
                .axis_iter(Axis(0)),
        )
        .zip(priors.anchors.iter())
        .zip(
            scores
                .view()
                .to_shape((num_boxes, 2))
                .unwrap()
                .axis_iter(Axis(0)),
        )
        .filter_map(|(((rect, landmarks), prior), score)| {
            let score = score[1];

            if score > 0.95 {
                let rect = priors.decode_box(prior, &(rect[0], rect[1], rect[2], rect[3]));
                let rect = rect.scale(scale_ratios.0, scale_ratios.1);

                let landmarks = landmarks
                    .to_vec()
                    .chunks(2)
                    .map(|point| {
                        let point = priors.decode_landmark(prior, (point[0], point[1]));
                        (point.0 * scale_ratios.0, point.1 * scale_ratios.1)
                    })
                    .collect::<Vec<_>>();

                Some(Face {
                    rect,
                    landmarks: Some(landmarks),
                    confidence: score,
                })
            } else {
                None
            }
        })
        .collect();

    Nms::default().suppress_non_maxima(faces)
}

fn convert_rect(rect: Rect) -> imageproc::rect::Rect {
    imageproc::rect::Rect::at(rect.x as i32, rect.y as i32)
        .of_size(rect.width as u32, rect.height as u32)
}

/// Draws faces on the image.
pub fn draw_faces<I>(image: &mut I, faces: Vec<Face>)
where
    I: GenericImage<Pixel = Rgb<u8>>,
{
    for face in faces {
        imageproc::drawing::draw_hollow_rect_mut(image, convert_rect(face.rect), Rgb([0, 255, 0]));
        for lm in face.landmarks.unwrap_or_default() {
            imageproc::drawing::draw_filled_circle_mut(
                image,
                (lm.0 as i32, lm.1 as i32),
                2,
                Rgb([255, 0, 0]),
            );
        }
    }
}
