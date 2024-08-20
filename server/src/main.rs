use opencv::{
    core::{Mat, Vector},
    imgcodecs,
    prelude::*,
    videoio,
};

use std::io::Write;
use std::net::TcpListener;

use zenoh::{session::SessionDeclarations, Config, Wait};

fn main() {
    let listener = TcpListener::bind("0.0.0.0:8080").unwrap();
    println!("Server listening on port 8080");

    let mut cam =
        videoio::VideoCapture::new(0, videoio::CAP_ANY).expect("Failed to get video capture");
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 1248.0).unwrap();
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 736.0).unwrap();
    let mut frame = Mat::default();
    let mut buf = Vector::new();

    let session = zenoh::open(Config::default()).wait().unwrap();

    let publisher = session.declare_publisher("video").wait().unwrap();

    loop {
        let (mut stream, _) = listener.accept().expect("Failed to accept connection");

        cam.read(&mut frame).expect("Failed to capture frame");
        buf.clear();
        let _ = imgcodecs::imencode(".jpg", &frame, &mut buf, &Vector::new());

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        );

        stream.write_all(response.as_bytes()).unwrap();

        loop {
            cam.read(&mut frame).expect("Failed to capture frame");
            buf.clear();
            
            let _ = imgcodecs::imencode(".jpg", &frame, &mut buf, &Vector::new());

            let image_data = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                buf.len()
            );

            stream.write_all(image_data.as_bytes()).unwrap();
            stream.write_all(buf.as_slice()).unwrap();
            stream.write_all(b"\r\n").unwrap();
            stream.flush().unwrap();

            publisher.put(frame.data_bytes().unwrap()).wait().unwrap();
        }
    }
}
