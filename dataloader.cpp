#include "../include/model.hpp"
#include <iostream>

namespace models {

// ============================================================================
// RESIDUAL BLOCK IMPLEMENTATION
// ============================================================================

ResidualBlockImpl::ResidualBlockImpl(int64_t num_features)
    : conv1_(torch::nn::Conv2dOptions(num_features, num_features, 3)
                 .padding(1)),
      bn1_(num_features),
      conv2_(torch::nn::Conv2dOptions(num_features, num_features, 3)
                 .padding(1)),
      bn2_(num_features) {
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);
    register_module("conv2", conv2_);
    register_module("bn2", bn2_);
}

torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& x) {
    auto out = torch::relu(bn1_(conv1_(x)));
    out = bn2_(conv2_(out));
    return x + out;  // Skip connection
}

// ============================================================================
// DOWNSAMPLING BLOCK IMPLEMENTATION
// ============================================================================

DownsampleBlockImpl::DownsampleBlockImpl(int64_t in_channels, int64_t out_channels)
    : conv_(torch::nn::Conv2dOptions(in_channels, out_channels, 4)
                .stride(2)
                .padding(1)),
      bn_(out_channels) {
    register_module("conv", conv_);
    register_module("bn", bn_);
}

torch::Tensor DownsampleBlockImpl::forward(const torch::Tensor& x) {
    return torch::leaky_relu(bn_(conv_(x)), 0.2);
}

// ============================================================================
// UPSAMPLING BLOCK IMPLEMENTATION
// ============================================================================

UpsampleBlockImpl::UpsampleBlockImpl(int64_t in_channels, int64_t out_channels, bool use_dropout)
    : deconv_(torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4)
                  .stride(2)
                  .padding(1)),
      bn_(out_channels),
      dropout_(use_dropout ? torch::nn::Dropout(0.5) : torch::nn::Dropout(0.0)) {
    register_module("deconv", deconv_);
    register_module("bn", bn_);
    if (use_dropout) {
        register_module("dropout", dropout_);
    }
}

torch::Tensor UpsampleBlockImpl::forward(const torch::Tensor& x) {
    auto out = torch::relu(bn_(deconv_(x)));
    if (dropout_->options.p() > 0) {
        out = dropout_(out);
    }
    return out;
}

// ============================================================================
// U-NET GENERATOR IMPLEMENTATION
// ============================================================================

UNetGeneratorImpl::UNetGeneratorImpl(int64_t input_channels,
                                   int64_t output_channels,
                                   int64_t num_filters)
    : conv0_(torch::nn::Conv2dOptions(input_channels, num_filters, 7)
                 .padding(3)) {
    register_module("conv0", conv0_);

    // Encoder (downsampling)
    int64_t current_filters = num_filters;
    for (int i = 0; i < 3; ++i) {
        encoder_.push_back(DownsampleBlock(current_filters, current_filters * 2));
        register_module("encoder_" + std::to_string(i),
                       encoder_[i]);
        current_filters *= 2;
    }

    // Bottleneck (residual blocks)
    for (int i = 0; i < 6; ++i) {
        bottleneck_.push_back(ResidualBlock(current_filters));
        register_module("bottleneck_" + std::to_string(i),
                       bottleneck_[i]);
    }

    // Decoder (upsampling) with skip connections
    for (int i = 0; i < 3; ++i) {
        current_filters /= 2;
        bool use_dropout = (i < 1);  // Dropout only in first upsample
        decoder_.push_back(UpsampleBlock(current_filters * 2, current_filters, use_dropout));
        register_module("decoder_" + std::to_string(i),
                       decoder_[i]);
    }

    // Final output layer (map back to 3 channels)
    conv_final_ = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(num_filters, output_channels, 7).padding(3));
    register_module("conv_final", conv_final_);
}

torch::Tensor UNetGeneratorImpl::forward(const torch::Tensor& x) {
    // Initial convolution
    auto conv0_out = torch::relu(conv0_(x));

    // Encoder with skip connections
    std::vector<torch::Tensor> skip_connections;
    auto enc_out = conv0_out;

    for (auto& encoder_block : encoder_) {
        skip_connections.push_back(enc_out);
        enc_out = encoder_block(enc_out);
    }

    // Bottleneck
    auto bottleneck_out = enc_out;
    for (auto& res_block : bottleneck_) {
        bottleneck_out = res_block(bottleneck_out);
    }

    // Decoder with skip connections
    auto dec_out = bottleneck_out;
    for (size_t i = 0; i < decoder_.size(); ++i) {
        dec_out = decoder_[i](dec_out);
        // Concatenate skip connection
        dec_out = torch::cat({dec_out, skip_connections[encoder_.size() - 1 - i]}, 1);
    }

    // Final output (tanh activation for [-1, 1] range)
    auto output = torch::tanh(conv_final_(dec_out));
    return output;
}

// ============================================================================
// PATCHGAN DISCRIMINATOR IMPLEMENTATION
// ============================================================================

PatchGANDiscriminatorImpl::PatchGANDiscriminatorImpl(int64_t input_channels,
                                                   int64_t num_filters) {
    // PatchGAN architecture: progressively downsample to patch-level discrimination
    model_ = torch::nn::Sequential(
        // Conv1: 256 -> 128
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 4)
                              .stride(2)
                              .padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

        // Conv2: 128 -> 64
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, num_filters * 2, 4)
                              .stride(2)
                              .padding(1)),
        torch::nn::BatchNorm2d(num_filters * 2),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

        // Conv3: 64 -> 32
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters * 2, num_filters * 4, 4)
                              .stride(2)
                              .padding(1)),
        torch::nn::BatchNorm2d(num_filters * 4),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

        // Conv4: 32 -> 16
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters * 4, num_filters * 8, 4)
                              .stride(1)
                              .padding(1)),
        torch::nn::BatchNorm2d(num_filters * 8),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

        // Final conv: outputs patch classification (1 channel)
        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters * 8, 1, 4)
                              .stride(1)
                              .padding(1))
    );

    register_module("model", model_);
}

torch::Tensor PatchGANDiscriminatorImpl::forward(const torch::Tensor& x) {
    return model_(x);
}

// ============================================================================
// CYCLEGAN IMPLEMENTATION
// ============================================================================

CycleGANImpl::CycleGANImpl(int64_t num_filters)
    : generator_A2B_(num_filters),
      generator_B2A_(num_filters),
      discriminator_A_(3, num_filters),
      discriminator_B_(3, num_filters) {
    register_module("generator_A2B", generator_A2B_);
    register_module("generator_B2A", generator_B2A_);
    register_module("discriminator_A", discriminator_A_);
    register_module("discriminator_B", discriminator_B_);
}

CycleGANImpl::ForwardOutput CycleGANImpl::forward(
    const torch::Tensor& real_A,
    const torch::Tensor& real_B) {

    // Generate fake images
    torch::Tensor fake_B = generator_A2B_(real_A);
    torch::Tensor fake_A = generator_B2A_(real_B);

    // Recover original images (cycle consistency)
    torch::Tensor recovered_A = generator_B2A_(fake_B);
    torch::Tensor recovered_B = generator_A2B_(fake_A);

    // Identity mapping (consistency if image is already in target domain)
    torch::Tensor identity_A = generator_B2A_(real_A);
    torch::Tensor identity_B = generator_A2B_(real_B);

    return ForwardOutput{fake_B, fake_A, recovered_A, recovered_B,
                         identity_A, identity_B};
}

torch::Tensor CycleGANImpl::restore(const torch::Tensor& murky_image) {
    // Inference: restore murky (A) to clear (B)
    return generator_A2B_(murky_image);
}

}  // namespace models
