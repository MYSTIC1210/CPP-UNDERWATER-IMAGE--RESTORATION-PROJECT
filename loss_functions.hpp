#pragma once

#include <torch/torch.h>
#include <memory>

namespace models {

// ============================================================================
// RESIDUAL BLOCK (for U-Net generator)
// ============================================================================
class ResidualBlockImpl : public torch::nn::Module {
 public:
    explicit ResidualBlockImpl(int64_t num_features);

    torch::Tensor forward(const torch::Tensor& x);

 private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::BatchNorm2d bn2_{nullptr};
};
TORCH_MODULE(ResidualBlock);

// ============================================================================
// DOWNSAMPLING BLOCK (Conv + BatchNorm + LeakyReLU)
// ============================================================================
class DownsampleBlockImpl : public torch::nn::Module {
 public:
    DownsampleBlockImpl(int64_t in_channels, int64_t out_channels);

    torch::Tensor forward(const torch::Tensor& x);

 private:
    torch::nn::Conv2d conv_{nullptr};
    torch::nn::BatchNorm2d bn_{nullptr};
};
TORCH_MODULE(DownsampleBlock);

// ============================================================================
// UPSAMPLING BLOCK (Deconv + BatchNorm + ReLU)
// ============================================================================
class UpsampleBlockImpl : public torch::nn::Module {
 public:
    UpsampleBlockImpl(int64_t in_channels, int64_t out_channels, bool use_dropout = false);

    torch::Tensor forward(const torch::Tensor& x);

 private:
    torch::nn::ConvTranspose2d deconv_{nullptr};
    torch::nn::BatchNorm2d bn_{nullptr};
    torch::nn::Dropout dropout_;
};
TORCH_MODULE(UpsampleBlock);

// ============================================================================
// U-NET GENERATOR (A -> B restoration)
// ============================================================================
class UNetGeneratorImpl : public torch::nn::Module {
 public:
    explicit UNetGeneratorImpl(int64_t input_channels = 3,
                              int64_t output_channels = 3,
                              int64_t num_filters = 64);

    torch::Tensor forward(const torch::Tensor& x);

 private:
    // Encoder
    torch::nn::Conv2d conv0_{nullptr};
    std::vector<DownsampleBlock> encoder_;

    // Bottleneck
    std::vector<ResidualBlock> bottleneck_;

    // Decoder with skip connections
    std::vector<UpsampleBlock> decoder_;

    // Output
    torch::nn::Conv2d conv_final_{nullptr};
};
TORCH_MODULE(UNetGenerator);

// ============================================================================
// PATCHGAN DISCRIMINATOR
// Classifies N x N patches of the image as real or fake
// ============================================================================
class PatchGANDiscriminatorImpl : public torch::nn::Module {
 public:
    explicit PatchGANDiscriminatorImpl(int64_t input_channels = 3,
                                      int64_t num_filters = 64);

    torch::Tensor forward(const torch::Tensor& x);

 private:
    torch::nn::Sequential model_{nullptr};
};
TORCH_MODULE(PatchGANDiscriminator);

// ============================================================================
// CYCLEgan MODEL (combines two generators and two discriminators)
// ============================================================================
class CycleGANImpl : public torch::nn::Module {
 public:
    explicit CycleGANImpl(int64_t num_filters = 64);

    // Forward passes for training
    struct ForwardOutput {
        torch::Tensor fake_B;
        torch::Tensor fake_A;
        torch::Tensor recovered_A;
        torch::Tensor recovered_B;
        torch::Tensor identity_A;
        torch::Tensor identity_B;
    };

    ForwardOutput forward(const torch::Tensor& real_A,
                          const torch::Tensor& real_B);

    // Inference: restore murky image (A) to clear image (B)
    torch::Tensor restore(const torch::Tensor& murky_image);

    // Get generator and discriminator references for training
    UNetGenerator& get_generator_A2B() { return *generator_A2B_; }
    UNetGenerator& get_generator_B2A() { return *generator_B2A_; }
    PatchGANDiscriminator& get_discriminator_A() { return *discriminator_A_; }
    PatchGANDiscriminator& get_discriminator_B() { return *discriminator_B_; }

 private:
    UNetGenerator generator_A2B_{nullptr};  // Murky -> Clear
    UNetGenerator generator_B2A_{nullptr};  // Clear -> Murky

    PatchGANDiscriminator discriminator_A_{nullptr};  // Discriminates murky images
    PatchGANDiscriminator discriminator_B_{nullptr};  // Discriminates clear images
};
TORCH_MODULE(CycleGAN);

}  // namespace models
