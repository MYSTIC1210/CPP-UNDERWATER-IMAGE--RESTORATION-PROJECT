#include "../include/dataloader.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace data {

// ============================================================================
// IMAGE PREPROCESSOR IMPLEMENTATION
// ============================================================================

ImagePreprocessor::ImagePreprocessor(int height, int width)
    : target_height_(height), target_width_(width) {}

torch::Tensor ImagePreprocessor::load_image(const std::string& image_path) {
    // Load image using OpenCV
    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // Resize to target dimensions
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_width_, target_height_),
               0, 0, cv::INTER_LINEAR);

    // Convert to tensor
    return mat_to_tensor(resized);
}

torch::Tensor ImagePreprocessor::mat_to_tensor(const cv::Mat& mat) {
    // Convert OpenCV BGR image to tensor [C, H, W]
    // Input: [H, W, 3] BGR
    // Output: [3, H, W] normalized to [-1, 1]

    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F, 1.0 / 255.0);  // Normalize to [0, 1]

    // Convert BGR to RGB
    cv::Mat rgb_mat;
    cv::cvtColor(float_mat, rgb_mat, cv::COLOR_BGR2RGB);

    // Create tensor from Mat
    torch::Tensor tensor = torch::from_blob(
        rgb_mat.data,
        {1, rgb_mat.rows, rgb_mat.cols, 3},
        torch::kFloat32);

    // Permute to [1, 3, H, W]
    tensor = tensor.permute({0, 3, 1, 2});

    // Normalize to [-1, 1]
    tensor = (tensor - 0.5f) * 2.0f;

    // Remove batch dimension
    return tensor.squeeze(0).clone();
}

cv::Mat ImagePreprocessor::tensor_to_mat(const torch::Tensor& tensor) {
    // Convert tensor [3, H, W] in [-1, 1] to OpenCV Mat [H, W, 3] in [0, 255]

    // Denormalize to [0, 1]
    torch::Tensor normalized = (tensor + 1.0f) / 2.0f;

    // Clamp to valid range
    normalized = torch::clamp(normalized, 0.0f, 1.0f);

    // Permute to [H, W, 3]
    torch::Tensor permuted = normalized.permute({1, 2, 0});

    // Convert to CPU and get data pointer
    auto tensor_cpu = permuted.detach().cpu();
    float* data = tensor_cpu.data_ptr<float>();

    // Create OpenCV Mat from tensor
    cv::Mat mat(tensor_cpu.size(0), tensor_cpu.size(1), CV_32FC3, data);

    // Convert RGB to BGR
    cv::Mat bgr_mat;
    cv::cvtColor(mat, bgr_mat, cv::COLOR_RGB2BGR);

    // Convert to uint8 [0, 255]
    cv::Mat uint8_mat;
    bgr_mat.convertTo(uint8_mat, CV_8U, 255.0);

    return uint8_mat.clone();
}

cv::Mat ImagePreprocessor::bgr_to_lab(const cv::Mat& bgr_image) {
    cv::Mat lab;
    cv::cvtColor(bgr_image, lab, cv::COLOR_BGR2Lab);
    return lab;
}

cv::Mat ImagePreprocessor::lab_to_bgr(const cv::Mat& lab_image) {
    cv::Mat bgr;
    cv::cvtColor(lab_image, bgr, cv::COLOR_Lab2BGR);
    return bgr;
}

// ============================================================================
// EUVP PAIRED DATASET IMPLEMENTATION
// ============================================================================

EUVPDatasetImpl::EUVPDatasetImpl(const std::string& murky_dir,
                               const std::string& reference_dir,
                               int image_height,
                               int image_width,
                               bool augment)
    : murky_dir_(murky_dir),
      reference_dir_(reference_dir),
      preprocessor_(image_height, image_width),
      augment_(augment),
      generator_(std::random_device{}()) {

    // Scan for image files in murky directory
    image_names_ = scan_image_directory(murky_dir);

    if (image_names_.empty()) {
        throw std::runtime_error("No images found in directory: " + murky_dir);
    }

    std::cout << "[Dataset] Found " << image_names_.size()
              << " paired images in " << murky_dir << std::endl;
}

EUVPDatasetImpl::ImagePair EUVPDatasetImpl::get(size_t index) {
    const std::string& image_name = image_names_[index];

    // Load murky and reference images
    std::string murky_path = murky_dir_ + "/" + image_name;
    std::string reference_path = reference_dir_ + "/" + image_name;

    torch::Tensor murky = preprocessor_.load_image(murky_path);
    torch::Tensor reference = preprocessor_.load_image(reference_path);

    // Apply augmentation if enabled
    if (augment_) {
        murky = augment_image(murky);
        reference = augment_image(reference);
    }

    return {murky, reference};
}

torch::optional<size_t> EUVPDatasetImpl::size() const {
    return image_names_.size();
}

const std::vector<std::string>& EUVPDatasetImpl::get_image_names() const {
    return image_names_;
}

torch::Tensor EUVPDatasetImpl::augment_image(const torch::Tensor& img) {
    std::uniform_real_distribution<> uniform(0.0, 1.0);

    auto result = img.clone();

    // Random horizontal flip (50%)
    if (uniform(generator_) > 0.5) {
        result = torch::flip(result, {-1});
    }

    // Random vertical flip (30%)
    if (uniform(generator_) > 0.7) {
        result = torch::flip(result, {-2});
    }

    return result;
}

// ============================================================================
// UNPAIRED DATASET IMPLEMENTATION
// ============================================================================

UnpairedEUVPDatasetImpl::UnpairedEUVPDatasetImpl(
    const std::string& murky_dir,
    const std::string& reference_dir,
    int image_height,
    int image_width)
    : preprocessor_(image_height, image_width) {

    murky_images_ = scan_image_directory(murky_dir);
    reference_images_ = scan_image_directory(reference_dir);

    if (murky_images_.empty() || reference_images_.empty()) {
        throw std::runtime_error("No images found in one of the directories");
    }

    std::cout << "[UnpairedDataset] Murky: " << murky_images_.size()
              << " | Reference: " << reference_images_.size() << std::endl;
}

UnpairedEUVPDatasetImpl::UnpairedImages UnpairedEUVPDatasetImpl::get(size_t index) {
    // Unpaired: use index modulo to cycle through dataset
    size_t murky_idx = index % murky_images_.size();
    size_t reference_idx = index % reference_images_.size();

    torch::Tensor murky = preprocessor_.load_image(
        murky_images_[murky_idx]);
    torch::Tensor reference = preprocessor_.load_image(
        reference_images_[reference_idx]);

    return {murky, reference};
}

torch::optional<size_t> UnpairedEUVPDatasetImpl::size() const {
    return std::max(murky_images_.size(), reference_images_.size());
}

// ============================================================================
// DATA UTILITIES IMPLEMENTATION
// ============================================================================

std::vector<std::string> scan_image_directory(const std::string& directory) {
    std::vector<std::string> image_files;

    if (!fs::exists(directory)) {
        std::cerr << "Warning: Directory does not exist: " << directory << std::endl;
        return image_files;
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();

            // Convert to lowercase
            std::transform(extension.begin(), extension.end(),
                         extension.begin(), ::tolower);

            if (extension == ".jpg" || extension == ".jpeg" ||
                extension == ".png" || extension == ".bmp") {
                image_files.push_back(entry.path().filename().string());
            }
        }
    }

    // Sort for deterministic ordering
    std::sort(image_files.begin(), image_files.end());

    return image_files;
}

auto create_paired_dataloader(
    const std::string& murky_dir,
    const std::string& reference_dir,
    int batch_size,
    int image_height,
    int image_width,
    bool shuffle,
    int num_workers) -> torch::data::DataLoaderOptions {

    // This is a helper function showing the pattern
    // Actual usage would be:
    // auto dataset = std::make_shared<EUVPDatasetImpl>(murky_dir, reference_dir, ...);
    // auto dataloader = torch::data::make_data_loader(dataset, batch_size, shuffle);

    torch::data::DataLoaderOptions options(batch_size);
    if (shuffle) {
        options.enforce_ordering(false);
    }
    return options;
}

auto create_unpaired_dataloader(
    const std::string& murky_dir,
    const std::string& reference_dir,
    int batch_size,
    int image_height,
    int image_width,
    bool shuffle,
    int num_workers) -> torch::data::DataLoaderOptions {

    torch::data::DataLoaderOptions options(batch_size);
    if (shuffle) {
        options.enforce_ordering(false);
    }
    return options;
}

}  // namespace data
