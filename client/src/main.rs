use std::net::TcpListener;

use image::ImageBuffer;
use ndarray::{self, Array3, ArrayBase};
use rust_faces::{
    viz, BlazeFaceParams, FaceDetection, FaceDetectorBuilder, InferParams, Provider, ToRgb8,
};
use std::io::Write;
use zenoh::{session::SessionDeclarations, Config, Wait};

fn main() {
    tracing_subscriber::fmt::init();
    let session = zenoh::open(Config::default()).wait().unwrap();

    let face_detector =
        FaceDetectorBuilder::new(FaceDetection::BlazeFace320(BlazeFaceParams::default()))
            .download()
            .infer_params(InferParams {
                provider: Provider::OrtCuda(0),
                intra_threads: Some(5),
                ..Default::default()
            })
            .build()
            .expect("Fail to load the face detector.");

    let subscriber = session.declare_subscriber("video").wait().unwrap();

    let listener = TcpListener::bind("0.0.0.0:8080").unwrap();

    loop {
        let (mut stream, _) = listener.accept().expect("Failed to accept connection");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        );

        stream.write_all(response.as_bytes()).unwrap();

        while let Ok(sample) = subscriber.recv() {
            let payload: Vec<u8> = sample.payload().deserialize().unwrap();
            // let image = Mat::new_rows_cols_with_data(1248, 736, &payload).unwrap();
            let image = Array3::from_shape_vec([736, 1248, 3], payload).unwrap();
            let faces = face_detector.detect(image.view().into_dyn()).unwrap();

            let mut image = to_rbg8(&image);
            viz::draw_faces(&mut image, faces);

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
