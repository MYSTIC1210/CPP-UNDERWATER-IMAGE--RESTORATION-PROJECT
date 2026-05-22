#pragma once

#include <string>
#include <vector>

namespace config {

// ============================================================================
// IMAGE PROCESSING
// ============================================================================
constexpr int IMAGE_HEIGHT = 256;
constexpr int IMAGE_WIDTH = 256;
constexpr int IMAGE_CHANNELS = 3;
constexpr float PIXEL_NORM_VALUE = 255.0f;

// ============================================================================
// TRAINING HYPERPARAMETERS
// ============================================================================
constexpr int BATCH_SIZE = 4;
constexpr int NUM_EPOCHS = 200;
constexpr int SAVE_INTERVAL = 10;  // Save model every N epochs
constexpr float LEARNING_RATE = 0.0002f;
constexpr float BETA1 = 0.5f;       // Adam optimizer beta1
constexpr float BETA2 = 0.999f;     // Adam optimizer beta2

// ============================================================================
// LOSS WEIGHTS (Cycle Consistency Loss vs Adversarial Loss)
// ============================================================================
constexpr float LAMBDA_CYCLE = 10.0f;    // Weight for cycle consistency loss
constexpr float LAMBDA_IDENTITY = 5.0f;  // Weight for identity loss
constexpr float LAMBDA_ADV = 1.0f;       // Weight for adversarial loss

// ============================================================================
// MODEL ARCHITECTURE
// ============================================================================
constexpr int GENERATOR_FILTERS = 64;     // Base filter count for generator
constexpr int DISCRIMINATOR_FILTERS = 64; // Base filter count for discriminator
constexpr int PATCH_SIZE = 70;            // PatchGAN patch size

// ============================================================================
// DEVICE CONFIGURATION
// ============================================================================
const std::string DEVICE = "cuda";  // "cuda" for GPU, "cpu" for CPU
constexpr bool USE_GPU = true;      // Enable GPU acceleration if available

// ============================================================================
// DATA PATHS
// ============================================================================
const std::string DATASET_ROOT = "./data/EUVP";
const std::string MURKY_TRAIN_PATH = "./data/EUVP/train/murky";
const std::string REFERENCE_TRAIN_PATH = "./data/EUVP/train/reference";
const std::string MURKY_TEST_PATH = "./data/EUVP/test/murky";
const std::string REFERENCE_TEST_PATH = "./data/EUVP/test/reference";

const std::string CHECKPOINT_DIR = "./checkpoints";
const std::string OUTPUT_DIR = "./results";

// ============================================================================
// INFERENCE CONFIGURATION
// ============================================================================
const std::string MODEL_WEIGHTS_PATH = "./checkpoints/final_generator.pt";
const std::string INPUT_IMAGES_DIR = "./input_images";
const std::string RESTORED_IMAGES_DIR = "./output_images";

}  // namespace config
