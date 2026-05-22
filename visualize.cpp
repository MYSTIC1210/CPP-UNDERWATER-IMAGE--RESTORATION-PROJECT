#include "../include/model.hpp"
#include "../include/dataloader.hpp"
#include "../include/loss_functions.hpp"
#include "../include/config.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cmath>
#include <filesystem>

using namespace models;
using namespace data;
using namespace losses;
using namespace config;

// ============================================================================
// TRAINING STATE AND UTILITIES
// ============================================================================

struct TrainingMetrics {
    float loss_D_A = 0.0f;
    float loss_D_B = 0.0f;
    float loss_G_A2B = 0.0f;
    float loss_G_B2A = 0.0f;
    float loss_cycle = 0.0f;
    float loss_identity = 0.0f;

    void reset() {
        loss_D_A = loss_D_B = loss_G_A2B = loss_G_B2A = 0.0f;
        loss_cycle = loss_identity = 0.0f;
    }

    void print_header() {
        std::cout << std::setw(8) << "Epoch"
                  << std::setw(12) << "D_A Loss"
                  << std::setw(12) << "D_B Loss"
                  << std::setw(12) << "G_A2B Loss"
                  << std::setw(12) << "G_B2A Loss"
                  << std::setw(12) << "Cycle Loss"
                  << std::setw(12) << "Identity"
                  << std::setw(12) << "Total Loss"
                  << std::endl;
        std::cout << std::string(92, '-') << std::endl;
    }

    void print(int epoch, int total_epochs) {
        float total = loss_D_A + loss_D_B + loss_G_A2B + loss_G_B2A +
                     loss_cycle + loss_identity;
        std::cout << std::setw(8) << epoch << "/" << total_epochs
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << loss_D_A
                  << std::setw(12) << loss_D_B
                  << std::setw(12) << loss_G_A2B
                  << std::setw(12) << loss_G_B2A
                  << std::setw(12) << loss_cycle
                  << std::setw(12) << loss_identity
                  << std::setw(12) << total
                  << std::endl;
    }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void ensure_directory_exists(const std::string& path) {
    std::filesystem::create_directories(path);
}

void save_checkpoint(const CycleGAN& model, int epoch,
                     const std::string& checkpoint_dir) {
    ensure_directory_exists(checkpoint_dir);

    std::string gen_a2b_path = checkpoint_dir + "/generator_A2B_epoch_" +
                               std::to_string(epoch) + ".pt";
    std::string gen_b2a_path = checkpoint_dir + "/generator_B2A_epoch_" +
                               std::to_string(epoch) + ".pt";
    std::string dis_a_path = checkpoint_dir + "/discriminator_A_epoch_" +
                             std::to_string(epoch) + ".pt";
    std::string dis_b_path = checkpoint_dir + "/discriminator_B_epoch_" +
                             std::to_string(epoch) + ".pt";

    try {
        torch::save(model.get_generator_A2B(), gen_a2b_path);
        torch::save(model.get_generator_B2A(), gen_b2a_path);
        torch::save(model.get_discriminator_A(), dis_a_path);
        torch::save(model.get_discriminator_B(), dis_b_path);
        std::cout << "[Checkpoint] Saved at epoch " << epoch << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to save checkpoint: " << e.what() << std::endl;
    }
}

// ============================================================================
// MAIN TRAINING FUNCTION
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Underwater Image Restoration - CycleGAN Training        ║" << std::endl;
    std::cout << "║   LibTorch + OpenCV + CUDA                                ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // DEVICE SETUP
    // ========================================================================
    torch::Device device(DEVICE);
    bool cuda_available = torch::cuda::is_available();

    std::cout << "[Device] CUDA Available: " << (cuda_available ? "YES" : "NO") << std::endl;
    if (cuda_available) {
        std::cout << "[Device] CUDA Devices: " << torch::cuda::device_count() << std::endl;
        std::cout << "[Device] Current Device: " << torch::cuda::current_device() << std::endl;
        torch::cuda::set_device(0);
    }
    std::cout << std::endl;

    // ========================================================================
    // MODEL INITIALIZATION
    // ========================================================================
    std::cout << "[Model] Initializing CycleGAN..." << std::endl;
    CycleGAN cycle_gan(GENERATOR_FILTERS);
    cycle_gan->to(device);

    std::cout << "[Model] Generator A2B parameters: "
              << std::count_if(cycle_gan->parameters().begin(),
                              cycle_gan->parameters().end(),
                              [](const auto& p) { return p.requires_grad(); })
              << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // LOSS FUNCTIONS
    // ========================================================================
    std::cout << "[Loss] Initializing loss functions..." << std::endl;
    CycleConsistencyLoss cycle_loss(LAMBDA_CYCLE);
    IdentityLoss identity_loss(LAMBDA_IDENTITY);
    AdversarialLoss adv_loss;

    cycle_loss->to(device);
    identity_loss->to(device);
    adv_loss->to(device);
    std::cout << std::endl;

    // ========================================================================
    // OPTIMIZERS
    // ========================================================================
    std::cout << "[Optimizer] Setting up Adam optimizers..." << std::endl;
    torch::optim::Adam gen_optimizer(
        std::vector<std::vector<torch::Tensor>>{
            {cycle_gan->get_generator_A2B()->parameters().begin(),
             cycle_gan->get_generator_A2B()->parameters().end()},
            {cycle_gan->get_generator_B2A()->parameters().begin(),
             cycle_gan->get_generator_B2A()->parameters().end()}
        },
        torch::optim::AdamOptions(LEARNING_RATE)
            .betas(std::make_tuple(BETA1, BETA2)));

    torch::optim::Adam dis_A_optimizer(
        cycle_gan->get_discriminator_A()->parameters(),
        torch::optim::AdamOptions(LEARNING_RATE)
            .betas(std::make_tuple(BETA1, BETA2)));

    torch::optim::Adam dis_B_optimizer(
        cycle_gan->get_generator_B2A()->parameters(),
        torch::optim::AdamOptions(LEARNING_RATE)
            .betas(std::make_tuple(BETA1, BETA2)));
    std::cout << std::endl;

    // ========================================================================
    // DATASET SETUP
    // ========================================================================
    std::cout << "[Dataset] Loading EUVP training dataset..." << std::endl;
    try {
        auto dataset = std::make_shared<EUVPDatasetImpl>(
            MURKY_TRAIN_PATH,
            REFERENCE_TRAIN_PATH,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            true  // Enable augmentation
        );

        auto dataloader = torch::data::make_data_loader(
            dataset,
            torch::data::DataLoaderOptions(BATCH_SIZE)
                .enforce_ordering(false));

        std::cout << "[Dataset] Dataset size: " << dataset->size().value() << std::endl;
        std::cout << std::endl;

        // ====================================================================
        // TRAINING LOOP
        // ====================================================================
        TrainingMetrics metrics;
        metrics.print_header();

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
            metrics.reset();
            int batch_count = 0;

            for (auto batch : *dataloader) {
                torch::Tensor murky = batch[0].murky.to(device);
                torch::Tensor reference = batch[0].reference.to(device);

                // Forward pass
                auto output = cycle_gan->forward(murky, reference);

                // ============================================================
                // GENERATOR TRAINING
                // ============================================================
                gen_optimizer.zero_grad();

                // Adversarial loss
                torch::Tensor fake_B_pred = cycle_gan->get_discriminator_B_(output.fake_B);
                torch::Tensor fake_A_pred = cycle_gan->get_discriminator_A_(output.fake_A);

                torch::Tensor loss_gen_A2B = adv_loss->generator_loss(fake_B_pred);
                torch::Tensor loss_gen_B2A = adv_loss->generator_loss(fake_A_pred);

                // Cycle consistency loss
                torch::Tensor loss_cyc = cycle_loss->forward(
                    murky, output.fake_B, reference, output.fake_A);

                // Identity loss
                torch::Tensor loss_id = identity_loss->forward(
                    murky, output.identity_A, reference, output.identity_B);

                // Total generator loss
                torch::Tensor total_gen_loss = loss_gen_A2B + loss_gen_B2A +
                                               loss_cyc + loss_id;

                total_gen_loss.backward();
                gen_optimizer.step();

                metrics.loss_G_A2B += loss_gen_A2B.item<float>();
                metrics.loss_G_B2A += loss_gen_B2A.item<float>();
                metrics.loss_cycle += loss_cyc.item<float>();
                metrics.loss_identity += loss_id.item<float>();

                // ============================================================
                // DISCRIMINATOR A TRAINING
                // ============================================================
                dis_A_optimizer.zero_grad();

                torch::Tensor real_A_pred = cycle_gan->get_discriminator_A_(murky);
                torch::Tensor fake_A_pred_detach = output.fake_A.detach();
                torch::Tensor fake_A_pred_dis = cycle_gan->get_discriminator_A_(fake_A_pred_detach);

                torch::Tensor loss_dis_A = adv_loss->discriminator_loss(
                    real_A_pred, fake_A_pred_dis);

                loss_dis_A.backward();
                dis_A_optimizer.step();

                metrics.loss_D_A += loss_dis_A.item<float>();

                // ============================================================
                // DISCRIMINATOR B TRAINING
                // ============================================================
                dis_B_optimizer.zero_grad();

                torch::Tensor real_B_pred = cycle_gan->get_discriminator_B_(reference);
                torch::Tensor fake_B_pred_detach = output.fake_B.detach();
                torch::Tensor fake_B_pred_dis = cycle_gan->get_discriminator_B_(fake_B_pred_detach);

                torch::Tensor loss_dis_B = adv_loss->discriminator_loss(
                    real_B_pred, fake_B_pred_dis);

                loss_dis_B.backward();
                dis_B_optimizer.step();

                metrics.loss_D_B += loss_dis_B.item<float>();

                batch_count++;
            }

            // Average metrics over batches
            metrics.loss_D_A /= batch_count;
            metrics.loss_D_B /= batch_count;
            metrics.loss_G_A2B /= batch_count;
            metrics.loss_G_B2A /= batch_count;
            metrics.loss_cycle /= batch_count;
            metrics.loss_identity /= batch_count;

            metrics.print(epoch, NUM_EPOCHS);

            // Save checkpoint
            if (epoch % SAVE_INTERVAL == 0) {
                save_checkpoint(cycle_gan, epoch, CHECKPOINT_DIR);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time);

        std::cout << std::endl;
        std::cout << "[Training] Complete! Total time: " << duration.count()
                  << " seconds (" << duration.count() / 60.0 << " minutes)" << std::endl;

        // Save final models
        save_checkpoint(cycle_gan, NUM_EPOCHS, CHECKPOINT_DIR);

    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
