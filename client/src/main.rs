use ndarray::{self, Array3};
use rust_faces::{
    viz, BlazeFaceParams, FaceDetection, FaceDetectorBuilder, InferParams, Provider, ToRgb8,
};
use zenoh::{session::SessionDeclarations, Config, Wait};

fn main() {
    let session = zenoh::open(Config::default()).wait().unwrap();

    let face_detector =
        FaceDetectorBuilder::new(FaceDetection::BlazeFace640(BlazeFaceParams::default()))
            .download()
            .infer_params(InferParams {
                provider: Provider::OrtCpu,
                intra_threads: Some(5),
                ..Default::default()
            })
            .build()
            .expect("Fail to load the face detector.");

    let subscriber = session.declare_subscriber("video").wait().unwrap();

    while let Ok(sample) = subscriber.recv() {
        let payload: Vec<u8> = sample.payload().deserialize().unwrap();
        // let image = Mat::new_rows_cols_with_data(1248, 736, &payload).unwrap();
        let image = Array3::from_shape_vec([736, 1248, 3], payload).unwrap();
        let faces = face_detector.detect(image.view().into_dyn()).unwrap();

        let mut image = image.to_rgb8();
        viz::draw_faces(&mut image, faces);
        std::fs::create_dir_all("tests/output").expect("Can't create test output dir.");
        image
            .save("tests/output/should_have_smooth_design.jpg")
            .expect("Can't save test image.");
    }
}
