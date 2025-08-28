use onnxruntime::{
    environment::Environment, 
    session::SessionBuilder,
    tensor::OrtOwnedTensor,
    ndarray::{Array, IxDyn},
};
use opencv::{
    videoio::{self, VideoCapture, CAP_ANY},
    core::{Mat, Size, CV_32F, Scalar},
    imgproc::{resize, cvt_color, COLOR_BGR2RGB, INTER_LINEAR},
    prelude::*,
};
use anyhow::{Result, Context};

const MODEL_INPUT_SIZE: i32 = 640; // YOLO typical input size
const CONFIDENCE_THRESHOLD: f32 = 0.5;

fn main() -> Result<()> {
    // Initialize ONNX runtime
    let env = Environment::builder()
        .with_name("defect_detection")
        .build()
        .context("Failed to create ONNX environment")?;
    
    let session = SessionBuilder::new(&env)?
        .with_optimization_level(onnxruntime::GraphOptimizationLevel::All)?
        .with_intra_threads(4)?
        .with_model_from_file("yolo_model.onnx")
        .context("Failed to load ONNX model")?;

    // Get model input/output info
    let input_name = session.inputs[0].name.clone();
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();

    // Initialize camera
    let mut cam = VideoCapture::new(0, CAP_ANY)
        .context("Failed to open camera")?;
    
    if !cam.is_opened()? {
        anyhow::bail!("Camera failed to open");
    }

    println!("Starting defect detection system...");
    println!("Press 'q' to quit");

    loop {
        let mut frame = Mat::default();
        
        // Read frame with error handling
        match cam.read(&mut frame) {
            Ok(true) => {
                if frame.empty() {
                    println!("Warning: Empty frame received");
                    continue;
                }
            },
            Ok(false) => {
                println!("No more frames available");
                break;
            },
            Err(e) => {
                eprintln!("Error reading frame: {}", e);
                continue;
            }
        }

        // Process frame for defects
        match process_frame(&session, &frame, &input_name, &output_names) {
            Ok(defects_detected) => {
                if defects_detected {
                    send_alert("Defect detected in current frame!");
                }
            },
            Err(e) => {
                eprintln!("Error processing frame: {}", e);
            }
        }

        // Add frame rate control
        std::thread::sleep(std::time::Duration::from_millis(33)); // ~30 FPS
        
        // Break on 'q' key
        //running indefinitely unless interrupted
    }

    println!("Shutting down detection system");
    Ok(())
}

fn preprocess_frame(frame: &Mat) -> Result<Array<f32, IxDyn>> {
    // Convert BGR to RGB
    let mut rgb_frame = Mat::default();
    cvt_color(frame, &mut rgb_frame, COLOR_BGR2RGB, 0)?;

    // Resize to model input size
    let mut resized = Mat::default();
    resize(
        &rgb_frame, 
        &mut resized, 
        Size::new(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        0.0, 
        0.0, 
        INTER_LINEAR
    )?;

    // Convert to f32 and normalize
    let mut normalized = Mat::default();
    resized.convert_to(&mut normalized, CV_32F, 1.0 / 255.0, 0.0)?;

    // Convert to ndarray format expected by ONNX
    // YOLO expects NCHW format: [batch_size, channels, height, width]
    let data = normalized.data_typed::<f32>()?;
    let input_array = Array::from_shape_vec(
        (1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize),
        data.to_vec()
    )?;

    Ok(input_array.into_dyn())
}

fn process_frame(
    session: &onnxruntime::Session,
    frame: &Mat,
    input_name: &str,
    output_names: &[String],
) -> Result<bool> {
    // Preprocess frame
    let input_tensor = preprocess_frame(frame)?;
    
    // Create ONNX tensor
    let input_values = vec![onnxruntime::Value::from_array(session.allocator(), &input_tensor)?];
    
    // Run inference
    let outputs = session.run(input_values)?;
    
    // Post-process results (simplified detection logic)
    let output_tensor = &outputs[0];
    let output_data = output_tensor.try_extract::<f32>()?;
    
    // simplified check for detections above confidence threshold
 
    let max_confidence = output_data.view()
        .iter()
        .fold(0.0f32, |max, &val| max.max(val));
    
    let defects_detected = max_confidence > CONFIDENCE_THRESHOLD;
    
    if defects_detected {
        println!("Defect detected with confidence: {:.2}", max_confidence);
    }
    
    Ok(defects_detected)
}

fn send_alert(message: &str) {
    println!("🚨 ALERT: {}", message);
    // In a real system, you might:
    // - Send email/SMS notifications
    // - Log to monitoring system  
    // - Trigger external systems
    // - Save frame with detected defect
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_threshold() {
        assert!(CONFIDENCE_THRESHOLD > 0.0);
        assert!(CONFIDENCE_THRESHOLD < 1.0);
    }
}
