#include "../include/model.hpp"
#include "../include/dataloader.hpp"
#include "../include/config.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace models;
using namespace data;
using namespace config;

// ============================================================================
// IMAGE COMPARISON METRICS
// ============================================================================

float calculate_psnr(const cv::Mat& original, const cv::Mat& restored) {
    CV_Assert(original.depth() == restored.depth());

    cv::Mat diff;
    cv::absdiff(original, restored, diff);
    diff = diff.mul(diff);

    cv::Scalar mean_squared_error = cv::mean(diff);
    double mse = (mean_squared_error[0] + mean_squared_error[1] +
                  mean_squared_error[2]) / 3.0;

    if (mse == 0.0) {
        return 100.0f;  // Identical images
    }

    double max_pixel = 255.0;
    float psnr = 20.0f * std::log10(max_pixel / std::sqrt(mse));
    return psnr;
}

float calculate_ssim(const cv::Mat& img1, const cv::Mat& img2) {
    // Simplified SSIM calculation
    // For production, consider using a full SSIM implementation

    CV_Assert(img1.size() == img2.size());

    cv::Mat img1_f, img2_f;
    img1.convertTo(img1_f, CV_32F);
    img2.convertTo(img2_f, CV_32F);

    cv::Mat diff = img1_f - img2_f;
    cv::Mat diff_squared;
    cv::multiply(diff, diff, diff_squared);

    float mse = cv::mean(diff_squared)[0];
    if (mse == 0.0) return 1.0f;

    float max_val = 255.0f;
    float ssim_approx = 1.0f - (mse / (max_val * max_val));
    return std::max(0.0f, ssim_approx);
}

// ============================================================================
// BATCH INFERENCE
// ============================================================================

struct InferenceResults {
    std::string filename;
    std::string status;
    float inference_time;
    float psnr;
    float ssim;
};

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Underwater Image Restoration - Inference                ║" << std::endl;
    std::cout << "║   CycleGAN Restoration Engine                             ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // ARGUMENT PARSING
    // ========================================================================
    std::string input_dir = INPUT_IMAGES_DIR;
    std::string output_dir = RESTORED_IMAGES_DIR;
    std::string model_path = MODEL_WEIGHTS_PATH;

    if (argc >= 2) {
        input_dir = argv[1];
    }
    if (argc >= 3) {
        output_dir = argv[2];
    }
    if (argc >= 4) {
        model_path = argv[3];
    }

    std::cout << "[Config] Input directory: " << input_dir << std::endl;
    std::cout << "[Config] Output directory: " << output_dir << std::endl;
    std::cout << "[Config] Model weights: " << model_path << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // DEVICE SETUP
    // ========================================================================
    torch::Device device(DEVICE);
    bool cuda_available = torch::cuda::is_available();

    std::cout << "[Device] CUDA Available: " << (cuda_available ? "YES" : "NO") << std::endl;
    if (cuda_available) {
        std::cout << "[Device] CUDA Devices: " << torch::cuda::device_count() << std::endl;
        torch::cuda::set_device(0);
    } else {
        std::cout << "[Device] Falling back to CPU" << std::endl;
        device = torch::Device(torch::kCPU);
    }
    std::cout << std::endl;

    try {
        // ====================================================================
        // MODEL LOADING
        // ====================================================================
        std::cout << "[Model] Loading CycleGAN generator..." << std::endl;

        CycleGAN cycle_gan(GENERATOR_FILTERS);

        if (std::filesystem::exists(model_path)) {
            try {
                torch::load(cycle_gan->get_generator_A2B(), model_path);
                std::cout << "[Model] Successfully loaded weights from: " << model_path << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[Warning] Could not load pretrained weights: " << e.what() << std::endl;
                std::cout << "[Model] Using randomly initialized weights" << std::endl;
            }
        } else {
            std::cout << "[Warning] Model file not found: " << model_path << std::endl;
            std::cout << "[Model] Using randomly initialized weights" << std::endl;
        }

        cycle_gan->to(device);
        cycle_gan->eval();  // Set to evaluation mode
        std::cout << std::endl;

        // ====================================================================
        // SCAN INPUT DIRECTORY
        // ====================================================================
        std::cout << "[I/O] Scanning input directory..." << std::endl;
        std::vector<std::string> image_files = scan_image_directory(input_dir);

        if (image_files.empty()) {
            std::cerr << "[Error] No images found in: " << input_dir << std::endl;
            return 1;
        }

        std::cout << "[I/O] Found " << image_files.size() << " images" << std::endl;
        std::cout << std::endl;

        // Create output directory
        std::filesystem::create_directories(output_dir);

        // ====================================================================
        // PREPROCESSING SETUP
        // ====================================================================
        ImagePreprocessor preprocessor(IMAGE_HEIGHT, IMAGE_WIDTH);
        std::vector<InferenceResults> results;

        // ====================================================================
        // BATCH INFERENCE LOOP
        // ====================================================================
        std::cout << "[Inference] Processing images..." << std::endl;
        std::cout << std::setw(30) << "Image"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(10) << "Status" << std::endl;
        std::cout << std::string(55, '-') << std::endl;

        for (const auto& image_name : image_files) {
            std::string input_path = input_dir + "/" + image_name;
            std::string output_path = output_dir + "/" + image_name;

            try {
                // Load image
                torch::Tensor murky_tensor = preprocessor.load_image(input_path);

                // Add batch dimension
                murky_tensor = murky_tensor.unsqueeze(0).to(device);

                // Inference
                auto start = std::chrono::high_resolution_clock::now();

                torch::NoGradGuard no_grad;  // Disable gradient computation
                torch::Tensor restored_tensor = cycle_gan->restore(murky_tensor);

                auto end = std::chrono::high_resolution_clock::now();
                float inference_time = std::chrono::duration<float, std::milli>(
                    end - start).count();

                // Remove batch dimension
                restored_tensor = restored_tensor.squeeze(0);

                // Convert tensor to OpenCV Mat
                cv::Mat restored_image = preprocessor.tensor_to_mat(restored_tensor);

                // Save restored image
                cv::imwrite(output_path, restored_image);

                InferenceResults result{
                    image_name,
                    "SUCCESS",
                    inference_time,
                    0.0f,  // PSNR not computed without reference
                    0.0f   // SSIM not computed without reference
                };
                results.push_back(result);

                std::cout << std::setw(30) << image_name
                          << std::setw(15) << std::fixed << std::setprecision(2)
                          << inference_time
                          << std::setw(10) << "✓" << std::endl;

            } catch (const std::exception& e) {
                InferenceResults result{
                    image_name,
                    std::string("FAILED: ") + e.what(),
                    0.0f,
                    0.0f,
                    0.0f
                };
                results.push_back(result);

                std::cout << std::setw(30) << image_name
                          << std::setw(15) << "-"
                          << std::setw(10) << "✗" << std::endl;
            }
        }

        // ====================================================================
        // SUMMARY REPORT
        // ====================================================================
        std::cout << std::endl;
        std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║   Inference Summary                                        ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << std::endl;

        int successful = 0;
        float total_time = 0.0f;
        float avg_time = 0.0f;

        for (const auto& result : results) {
            if (result.status == "SUCCESS") {
                successful++;
                total_time += result.inference_time;
            }
        }

        if (successful > 0) {
            avg_time = total_time / successful;
        }

        std::cout << "[Summary] Processed: " << successful << " / "
                  << image_files.size() << " images successfully" << std::endl;
        std::cout << "[Summary] Total time: " << std::fixed << std::setprecision(2)
                  << total_time << " ms" << std::endl;
        std::cout << "[Summary] Average time per image: " << avg_time << " ms" << std::endl;
        std::cout << "[Summary] Output directory: " << output_dir << std::endl;
        std::cout << std::endl;

        // Save results to CSV
        std::string results_file = output_dir + "/inference_results.csv";
        std::ofstream csv(results_file);
        csv << "Image,Status,InferenceTime_ms,PSNR,SSIM\n";
        for (const auto& result : results) {
            csv << result.filename << ","
                << result.status << ","
                << result.inference_time << ","
                << result.psnr << ","
                << result.ssim << "\n";
        }
        csv.close();

        std::cout << "[I/O] Results saved to: " << results_file << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Fatal Error] " << e.what() << std::endl;
        return 1;
    }
}
