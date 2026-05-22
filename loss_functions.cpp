#include "../include/model.hpp"
#include "../include/dataloader.hpp"
#include "../include/config.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <filesystem>

using namespace models;
using namespace data;
using namespace config;

// ============================================================================
// EVALUATION METRICS
// ============================================================================

struct ImageMetrics {
    std::string image_name;
    float psnr;
    float ssim;
    float mse;
    float mae;
    float inference_time;
};

class MetricsComputer {
 public:
    /**
     * Compute Peak Signal-to-Noise Ratio (PSNR)
     * Higher is better (>25 dB is good)
     */
    static float compute_psnr(const torch::Tensor& original,
                             const torch::Tensor& restored) {
        torch::Tensor diff = original - restored;
        torch::Tensor mse = torch::mean(diff * diff);

        if (mse.item<float>() == 0.0f) {
            return 100.0f;  // Identical images
        }

        float max_pixel = 1.0f;  // Tensors normalized to [0,1]
        float psnr = 20.0f * std::log10(max_pixel / std::sqrt(mse.item<float>()));
        return psnr;
    }

    /**
     * Compute Structural Similarity Index (SSIM)
     * Range: [-1, 1], higher is better (>0.7 is good)
     */
    static float compute_ssim(const torch::Tensor& original,
                             const torch::Tensor& restored) {
        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Mean
        torch::Tensor mu1 = original;
        torch::Tensor mu2 = restored;

        // Variance and covariance
        torch::Tensor mu1_sq = mu1 * mu1;
        torch::Tensor mu2_sq = mu2 * mu2;
        torch::Tensor mu1_mu2 = mu1 * mu2;

        torch::Tensor sigma1_sq = (original * original) - mu1_sq;
        torch::Tensor sigma2_sq = (restored * restored) - mu2_sq;
        torch::Tensor sigma12 = (original * restored) - mu1_mu2;

        // SSIM formula
        torch::Tensor numerator = (2.0f * mu1_mu2 + C1) * (2.0f * sigma12 + C2);
        torch::Tensor denominator = (mu1_sq + mu2_sq + C1) *
                                    (sigma1_sq + sigma2_sq + C2);

        torch::Tensor ssim = numerator / (denominator + 1e-8);
        return torch::mean(ssim).item<float>();
    }

    /**
     * Compute Mean Squared Error (MSE)
     * Lower is better
     */
    static float compute_mse(const torch::Tensor& original,
                            const torch::Tensor& restored) {
        torch::Tensor diff = original - restored;
        return torch::mean(diff * diff).item<float>();
    }

    /**
     * Compute Mean Absolute Error (MAE)
     * Lower is better (L1 distance)
     */
    static float compute_mae(const torch::Tensor& original,
                            const torch::Tensor& restored) {
        torch::Tensor diff = torch::abs(original - restored);
        return torch::mean(diff).item<float>();
    }
};

// ============================================================================
// EVALUATION ENGINE
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Underwater Image Restoration - Model Evaluation          ║" << std::endl;
    std::cout << "║   PSNR, SSIM, MSE, MAE Analysis                            ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // ARGUMENTS
    // ========================================================================
    if (argc < 4) {
        std::cerr << "Usage: evaluate <murky_dir> <reference_dir> <model_weights> [output_csv]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  murky_dir      Directory with murky (input) images" << std::endl;
        std::cerr << "  reference_dir  Directory with reference (ground truth) images" << std::endl;
        std::cerr << "  model_weights  Path to trained generator weights" << std::endl;
        std::cerr << "  output_csv     Optional CSV output file" << std::endl;
        return 1;
    }

    std::string murky_dir = argv[1];
    std::string reference_dir = argv[2];
    std::string model_path = argv[3];
    std::string output_csv = (argc >= 5) ? argv[4] : "evaluation_results.csv";

    // ========================================================================
    // DEVICE SETUP
    // ========================================================================
    torch::Device device(DEVICE);
    bool cuda_available = torch::cuda::is_available();

    std::cout << "[Device] CUDA Available: " << (cuda_available ? "YES" : "NO") << std::endl;
    if (!cuda_available && DEVICE == "cuda") {
        device = torch::Device(torch::kCPU);
        std::cout << "[Device] Falling back to CPU" << std::endl;
    }
    std::cout << std::endl;

    try {
        // ====================================================================
        // MODEL LOADING
        // ====================================================================
        std::cout << "[Model] Loading generator..." << std::endl;
        CycleGAN cycle_gan(GENERATOR_FILTERS);

        try {
            torch::load(cycle_gan->get_generator_A2B(), model_path);
            std::cout << "[Model] Weights loaded: " << model_path << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[Warning] Could not load weights: " << e.what() << std::endl;
            std::cout << "[Model] Using random initialization" << std::endl;
        }

        cycle_gan->to(device);
        cycle_gan->eval();
        std::cout << std::endl;

        // ====================================================================
        // SCAN DIRECTORIES
        // ====================================================================
        std::cout << "[I/O] Scanning directories..." << std::endl;
        std::vector<std::string> murky_files = scan_image_directory(murky_dir);
        std::vector<std::string> reference_files = scan_image_directory(reference_dir);

        std::cout << "[I/O] Murky images: " << murky_files.size() << std::endl;
        std::cout << "[I/O] Reference images: " << reference_files.size() << std::endl;

        if (murky_files.empty()) {
            std::cerr << "[Error] No murky images found" << std::endl;
            return 1;
        }

        std::cout << std::endl;

        // ====================================================================
        // EVALUATION LOOP
        // ====================================================================
        ImagePreprocessor preprocessor(IMAGE_HEIGHT, IMAGE_WIDTH);
        std::vector<ImageMetrics> all_metrics;

        std::cout << "[Evaluation] Computing metrics..." << std::endl;
        std::cout << std::setw(30) << "Image"
                  << std::setw(12) << "PSNR (dB)"
                  << std::setw(12) << "SSIM"
                  << std::setw(12) << "MSE"
                  << std::setw(12) << "MAE"
                  << std::setw(12) << "Time (ms)" << std::endl;
        std::cout << std::string(90, '-') << std::endl;

        int processed = 0;
        for (const auto& murky_name : murky_files) {
            // Find corresponding reference image
            std::string reference_name = murky_name;
            std::string reference_path = reference_dir + "/" + reference_name;
            std::string murky_path = murky_dir + "/" + murky_name;

            // Check if reference exists
            if (!std::filesystem::exists(reference_path)) {
                continue;  // Skip if no reference
            }

            try {
                // Load images
                torch::Tensor murky_tensor = preprocessor.load_image(murky_path);
                torch::Tensor reference_tensor = preprocessor.load_image(reference_path);

                // Add batch dimension
                murky_tensor = murky_tensor.unsqueeze(0).to(device);
                reference_tensor = reference_tensor.unsqueeze(0).to(device);

                // Inference
                auto start = std::chrono::high_resolution_clock::now();

                torch::NoGradGuard no_grad;
                torch::Tensor restored_tensor = cycle_gan->restore(murky_tensor);

                auto end = std::chrono::high_resolution_clock::now();
                float inference_time = std::chrono::duration<float, std::milli>(
                    end - start).count();

                // Remove batch dimension
                restored_tensor = restored_tensor.squeeze(0);
                reference_tensor = reference_tensor.squeeze(0);

                // Compute metrics
                float psnr = MetricsComputer::compute_psnr(reference_tensor, restored_tensor);
                float ssim = MetricsComputer::compute_ssim(reference_tensor, restored_tensor);
                float mse = MetricsComputer::compute_mse(reference_tensor, restored_tensor);
                float mae = MetricsComputer::compute_mae(reference_tensor, restored_tensor);

                ImageMetrics metrics{
                    murky_name, psnr, ssim, mse, mae, inference_time
                };
                all_metrics.push_back(metrics);

                std::cout << std::setw(30) << murky_name
                          << std::fixed << std::setprecision(2)
                          << std::setw(12) << psnr
                          << std::setw(12) << ssim
                          << std::setw(12) << mse
                          << std::setw(12) << mae
                          << std::setw(12) << inference_time << std::endl;

                processed++;

            } catch (const std::exception& e) {
                std::cerr << "[Warning] Failed to process " << murky_name
                         << ": " << e.what() << std::endl;
            }
        }

        std::cout << std::endl;

        // ====================================================================
        // STATISTICS
        // ====================================================================
        if (all_metrics.empty()) {
            std::cerr << "[Error] No images evaluated" << std::endl;
            return 1;
        }

        // Compute statistics
        float avg_psnr = 0.0f, avg_ssim = 0.0f, avg_mse = 0.0f, avg_mae = 0.0f;
        float max_psnr = -1e9f, min_psnr = 1e9f;
        float max_ssim = -1.0f, min_ssim = 1.0f;

        for (const auto& m : all_metrics) {
            avg_psnr += m.psnr;
            avg_ssim += m.ssim;
            avg_mse += m.mse;
            avg_mae += m.mae;

            max_psnr = std::max(max_psnr, m.psnr);
            min_psnr = std::min(min_psnr, m.psnr);
            max_ssim = std::max(max_ssim, m.ssim);
            min_ssim = std::min(min_ssim, m.ssim);
        }

        avg_psnr /= all_metrics.size();
        avg_ssim /= all_metrics.size();
        avg_mse /= all_metrics.size();
        avg_mae /= all_metrics.size();

        // Print summary
        std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║   Evaluation Summary                                        ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << std::endl;

        std::cout << "[Stats] Images Evaluated: " << all_metrics.size() << std::endl;
        std::cout << std::endl;

        std::cout << "[PSNR] Peak Signal-to-Noise Ratio (dB)" << std::endl;
        std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg_psnr << std::endl;
        std::cout << "  Max:     " << max_psnr << std::endl;
        std::cout << "  Min:     " << min_psnr << std::endl;
        std::cout << "  (Higher is better, >25 dB is excellent)" << std::endl;
        std::cout << std::endl;

        std::cout << "[SSIM] Structural Similarity Index" << std::endl;
        std::cout << "  Average: " << avg_ssim << std::endl;
        std::cout << "  Max:     " << max_ssim << std::endl;
        std::cout << "  Min:     " << min_ssim << std::endl;
        std::cout << "  (Range [-1,1], >0.7 is good)" << std::endl;
        std::cout << std::endl;

        std::cout << "[MSE] Mean Squared Error" << std::endl;
        std::cout << "  Average: " << avg_mse << std::endl;
        std::cout << "  (Lower is better)" << std::endl;
        std::cout << std::endl;

        std::cout << "[MAE] Mean Absolute Error" << std::endl;
        std::cout << "  Average: " << avg_mae << std::endl;
        std::cout << "  (Lower is better)" << std::endl;
        std::cout << std::endl;

        // ====================================================================
        // SAVE RESULTS
        // ====================================================================
        std::cout << "[I/O] Saving results to CSV..." << std::endl;
        std::ofstream csv(output_csv);
        csv << "Image,PSNR_dB,SSIM,MSE,MAE,InferenceTime_ms\n";

        for (const auto& m : all_metrics) {
            csv << m.image_name << ","
                << m.psnr << ","
                << m.ssim << ","
                << m.mse << ","
                << m.mae << ","
                << m.inference_time << "\n";
        }

        // Write summary statistics
        csv << "\n,Summary Statistics,,,,\n";
        csv << "Metric,Average,Max,Min,,\n";
        csv << "PSNR_dB," << avg_psnr << "," << max_psnr << "," << min_psnr << ",,\n";
        csv << "SSIM," << avg_ssim << "," << max_ssim << "," << min_ssim << ",,\n";
        csv << "MSE," << avg_mse << ",,,,\n";
        csv << "MAE," << avg_mae << ",,,,\n";

        csv.close();
        std::cout << "[I/O] Results saved: " << output_csv << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Fatal Error] " << e.what() << std::endl;
        return 1;
    }
}
