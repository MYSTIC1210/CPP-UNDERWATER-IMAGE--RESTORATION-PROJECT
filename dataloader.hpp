#pragma once

#include <torch/torch.h>

namespace losses {

/**
 * @brief Cycle Consistency Loss
 * Ensures that X -> Y -> X reconstruction is close to original X
 * Used in both directions: A->B->A and B->A->B
 */
class CycleConsistencyLoss : public torch::nn::Module {
 public:
    explicit CycleConsistencyLoss(float lambda = 10.0f);

    torch::Tensor forward(const torch::Tensor& real_A,
                         const torch::Tensor& fake_B,
                         const torch::Tensor& real_B,
                         const torch::Tensor& fake_A);

 private:
    float lambda_;
    torch::nn::L1Loss l1_loss_{torch::nn::L1LossOptions().reduction(torch::kMean)};
};

/**
 * @brief Identity Loss
 * Encourages G(X) ≈ X for images already in the target domain
 * Prevents unnecessary color/texture changes
 */
class IdentityLoss : public torch::nn::Module {
 public:
    explicit IdentityLoss(float lambda = 5.0f);

    torch::Tensor forward(const torch::Tensor& real_A,
                         const torch::Tensor& identity_A,
                         const torch::Tensor& real_B,
                         const torch::Tensor& identity_B);

 private:
    float lambda_;
    torch::nn::L1Loss l1_loss_{torch::nn::L1LossOptions().reduction(torch::kMean)};
};

/**
 * @brief Adversarial Loss (MSE-based GAN loss)
 * Computes loss between discriminator output and target (real=1, fake=0)
 */
class AdversarialLoss : public torch::nn::Module {
 public:
    AdversarialLoss() = default;

    // Generator loss: fools discriminator (wants output close to 1)
    torch::Tensor generator_loss(const torch::Tensor& discriminator_output);

    // Discriminator loss: distinguishes real from fake
    torch::Tensor discriminator_loss(const torch::Tensor& real_output,
                                      const torch::Tensor& fake_output);

 private:
    torch::nn::MSELoss mse_loss_{torch::nn::MSELossOptions().reduction(torch::kMean)};
};

/**
 * @brief Image Gradient Loss (Perceptual smoothness)
 * Encourages smooth gradients in the restored image
 */
class GradientLoss : public torch::nn::Module {
 public:
    explicit GradientLoss(float lambda = 1.0f);

    torch::Tensor forward(const torch::Tensor& image);

 private:
    float lambda_;
};

}  // namespace losses
