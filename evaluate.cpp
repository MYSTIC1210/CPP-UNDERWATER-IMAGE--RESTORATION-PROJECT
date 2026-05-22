#include "../include/dataloader.hpp"
#include "../include/config.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace data;
using namespace config;

namespace fs = std::filesystem;

// ============================================================================
// IMAGE COMPARISON UTILITIES
// ============================================================================

class ImageComparison {
 public:
    /**
     * Create side-by-side comparison image
     * @param murky Original murky image
     * @param restored Restored image
     * @param output_path Path to save comparison
     */
    static void create_comparison(const cv::Mat& murky,
                                  const cv::Mat& restored,
                                  const std::string& output_path) {
        // Ensure same size
        cv::Mat murky_resized = murky.clone();
        cv::Mat restored_resized = restored.clone();

        if (murky.size() != restored.size()) {
            cv::resize(restored, restored_resized, murky.size());
        }

        // Create side-by-side image
        cv::Mat comparison;
        cv::hconcat(murky_resized, restored_resized, comparison);

        // Add labels
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.8;
        int thickness = 2;
        cv::Scalar text_color(255, 255, 255);  // White text

        cv::putText(comparison, "Murky Input",
                   cv::Point(20, 40), font_face, font_scale, text_color, thickness);
        cv::putText(comparison, "Restored Output",
                   cv::Point(murky.cols + 20, 40), font_face, font_scale, text_color, thickness);

        // Save
        cv::imwrite(output_path, comparison);
        std::cout << "[Comparison] Saved: " << output_path << std::endl;
    }

    /**
     * Create difference map showing pixel-level changes
     * @param original Ground truth image
     * @param restored Restored image
     * @param output_path Path to save difference map
     */
    static void create_difference_map(const cv::Mat& original,
                                      const cv::Mat& restored,
                                      const std::string& output_path) {
        cv::Mat original_float, restored_float;
        original.convertTo(original_float, CV_32F, 1.0 / 255.0);
        restored.convertTo(restored_float, CV_32F, 1.0 / 255.0);

        // Compute absolute difference
        cv::Mat diff = cv::abs(original_float - restored_float);

        // Normalize to [0, 255]
        cv::Mat diff_normalized;
        cv::normalize(diff, diff_normalized, 0, 255, cv::NORM_MINMAX);
        diff_normalized.convertTo(diff_normalized, CV_8U);

        // Apply colormap (red = more difference)
        cv::Mat colored_diff;
        cv::applyColorMap(diff_normalized, colored_diff, cv::COLORMAP_HOT);

        // Save
        cv::imwrite(output_path, colored_diff);
        std::cout << "[Difference] Saved: " << output_path << std::endl;
    }

    /**
     * Create histogram comparison
     * @param original Ground truth image
     * @param restored Restored image
     * @param output_path Path to save histogram
     */
    static void create_histogram_comparison(const cv::Mat& original,
                                           const cv::Mat& restored,
                                           const std::string& output_path) {
        // Compute histograms
        int hist_size = 256;
        float range[] = {0, 256};
        const float* ranges = range;

        cv::Mat hist_original, hist_restored;
        cv::calcHist(&original, 1, 0, cv::Mat(), hist_original,
                     1, &hist_size, &ranges);
        cv::calcHist(&restored, 1, 0, cv::Mat(), hist_restored,
                     1, &hist_size, &ranges);

        // Normalize
        cv::normalize(hist_original, hist_original, 0, 255, cv::NORM_MINMAX);
        cv::normalize(hist_restored, hist_restored, 0, 255, cv::NORM_MINMAX);

        // Create visualization
        int hist_width = 512;
        int hist_height = 400;
        cv::Mat histogram_image(hist_height, hist_width, CV_8UC3, cv::Scalar(255, 255, 255));

        int bin_width = cvRound((double)hist_width / hist_size);

        // Draw original histogram (blue)
        for (int i = 1; i < hist_size; i++) {
            cv::line(histogram_image,
                    cv::Point(bin_width * (i - 1), hist_height - cvRound(hist_original.at<float>(i - 1))),
                    cv::Point(bin_width * i, hist_height - cvRound(hist_original.at<float>(i))),
                    cv::Scalar(255, 0, 0), 2);  // Blue
        }

        // Draw restored histogram (green)
        for (int i = 1; i < hist_size; i++) {
            cv::line(histogram_image,
                    cv::Point(bin_width * (i - 1), hist_height - cvRound(hist_restored.at<float>(i - 1))),
                    cv::Point(bin_width * i, hist_height - cvRound(hist_restored.at<float>(i))),
                    cv::Scalar(0, 255, 0), 2);  // Green
        }

        // Add legend
        cv::rectangle(histogram_image, cv::Point(10, 10), cv::Point(100, 50),
                     cv::Scalar(200, 200, 200), -1);
        cv::putText(histogram_image, "Original (Blue)", cv::Point(15, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        cv::putText(histogram_image, "Restored (Green)", cv::Point(15, 45),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        cv::imwrite(output_path, histogram_image);
        std::cout << "[Histogram] Saved: " << output_path << std::endl;
    }
};

// ============================================================================
// BATCH VISUALIZATION
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Underwater Image Comparison & Visualization Tool         ║" << std::endl;
    std::cout << "║   Side-by-side, Difference Maps, Histograms                ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // ARGUMENTS
    // ========================================================================
    if (argc < 4) {
        std::cerr << "Usage: visualize <murky_dir> <restored_dir> <output_dir> [reference_dir]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  murky_dir      Directory with original murky images" << std::endl;
        std::cerr << "  restored_dir   Directory with restored images" << std::endl;
        std::cerr << "  output_dir     Directory to save visualizations" << std::endl;
        std::cerr << "  reference_dir  Optional directory with ground truth images" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Creates:" << std::endl;
        std::cerr << "  *_comparison.jpg  - Side-by-side images" << std::endl;
        std::cerr << "  *_difference.jpg  - Pixel-level difference map" << std::endl;
        std::cerr << "  *_histogram.jpg   - Color histogram comparison" << std::endl;
        return 1;
    }

    std::string murky_dir = argv[1];
    std::string restored_dir = argv[2];
    std::string output_dir = argv[3];
    std::string reference_dir = (argc >= 5) ? argv[4] : "";

    // Create output directory
    fs::create_directories(output_dir);

    // ========================================================================
    // SCAN DIRECTORIES
    // ========================================================================
    std::cout << "[I/O] Scanning directories..." << std::endl;
    std::vector<std::string> murky_files = scan_image_directory(murky_dir);
    std::vector<std::string> restored_files = scan_image_directory(restored_dir);

    std::cout << "[I/O] Murky images: " << murky_files.size() << std::endl;
    std::cout << "[I/O] Restored images: " << restored_files.size() << std::endl;

    if (!reference_dir.empty()) {
        std::cout << "[I/O] Reference directory: " << reference_dir << std::endl;
    }
    std::cout << std::endl;

    if (murky_files.empty() || restored_files.empty()) {
        std::cerr << "[Error] No images found in input directories" << std::endl;
        return 1;
    }

    // ========================================================================
    // PROCESSING LOOP
    // ========================================================================
    std::cout << "[Processing] Creating visualizations..." << std::endl;
    std::cout << std::setw(30) << "Image"
              << std::setw(12) << "Comparison"
              << std::setw(12) << "Difference"
              << std::setw(12) << "Histogram" << std::endl;
    std::cout << std::string(66, '-') << std::endl;

    int processed = 0;
    for (const auto& restored_name : restored_files) {
        std::string murky_name = restored_name;
        std::string murky_path = murky_dir + "/" + murky_name;
        std::string restored_path = restored_dir + "/" + restored_name;

        // Check if murky image exists
        if (!fs::exists(murky_path)) {
            continue;
        }

        try {
            // Load images
            cv::Mat murky = cv::imread(murky_path);
            cv::Mat restored = cv::imread(restored_path);

            if (murky.empty() || restored.empty()) {
                std::cerr << "[Warning] Failed to load " << murky_name << std::endl;
                continue;
            }

            // Generate output filenames
            std::string base_name = murky_name.substr(0, murky_name.find_last_of('.'));
            std::string comparison_path = output_dir + "/" + base_name + "_comparison.jpg";
            std::string difference_path = output_dir + "/" + base_name + "_difference.jpg";
            std::string histogram_path = output_dir + "/" + base_name + "_histogram.jpg";

            // Create comparison
            ImageComparison::create_comparison(murky, restored, comparison_path);

            // Create difference map
            ImageComparison::create_difference_map(murky, restored, difference_path);

            // Create histogram
            ImageComparison::create_histogram_comparison(murky, restored, histogram_path);

            std::cout << std::setw(30) << murky_name
                      << std::setw(12) << "✓"
                      << std::setw(12) << "✓"
                      << std::setw(12) << "✓" << std::endl;

            processed++;

            // If reference available, also compare with it
            if (!reference_dir.empty()) {
                std::string reference_path = reference_dir + "/" + murky_name;
                if (fs::exists(reference_path)) {
                    cv::Mat reference = cv::imread(reference_path);
                    if (!reference.empty()) {
                        std::string ref_comparison = output_dir + "/" + base_name + "_vs_reference.jpg";
                        ImageComparison::create_comparison(reference, restored, ref_comparison);
                    }
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "[Error] " << murky_name << ": " << e.what() << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "[Summary] Processed " << processed << " images" << std::endl;
    std::cout << "[Output] Saved to: " << output_dir << std::endl;
    std::cout << "[Output] Total visualizations: " << (processed * 3) << std::endl;

    return 0;
}
