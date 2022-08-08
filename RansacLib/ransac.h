// Copyright (c) 2019, Torsten Sattler
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// author: Torsten Sattler, torsten.sattler.de@googlemail.com

#ifndef RANSACLIB_RANSACLIB_RANSAC_H_
#define RANSACLIB_RANSACLIB_RANSAC_H_

#define DEBUG true

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <RansacLib/sampling.h>
#include <RansacLib/utils.h>

namespace ransac_lib
{

  class RansacOptions
  {
  public:
    RansacOptions()
        : min_num_iterations_(100u),
          max_num_iterations_(10000u),
          success_probability_(0.9999),
          squared_inlier_threshold_(1.0),
          random_seed_(0u) {}
    uint32_t min_num_iterations_;
    uint32_t max_num_iterations_;
    double success_probability_;
    double squared_inlier_threshold_;
    unsigned int random_seed_;
  };

  // See Lebeda et al., Fixing the Locally Optimized RANSAC, BMVC, Table 1 for
  // details on the variables.
  class LORansacOptions : public RansacOptions
  {
  public:
    LORansacOptions()
        : num_lo_steps_(10),
          threshold_multiplier_(std::sqrt(2.0)),
          num_lsq_iterations_(4),
          min_sample_multiplicator_(7),
          non_min_sample_multiplier_(3),
          lo_starting_iterations_(50u),
          final_least_squares_(false) {}
    int num_lo_steps_;
    double threshold_multiplier_;
    int num_lsq_iterations_;
    // The maximum number of data points used for least squares refinement is
    // min_sample_multiplicator_ * min_sample_size. Lebeda et al. recommend
    // setting min_sample_multiplicator_ to 7 (empirically determined for
    // epipolar geometry estimation.
    int min_sample_multiplicator_;
    // The solver needs to report the minimal size of the non-minimal sample
    // needed for its non-minimal solver. In practice, we draw a sample of size
    // min(non_min_sample_size * non_min_sample_multiplier_, N / 2), where N is
    // the number of data points.
    int non_min_sample_multiplier_;
    // As suggested in Sec. 4.4 in Lebeda et al., Local Optimization is only
    // performed after the first K_start iterations (set to 50 by Lebeda et al.)
    // to reduce overhead.
    uint32_t lo_starting_iterations_;
    bool final_least_squares_;

//      / Once a model reaches this level of unlikelihood, it is rejected. Set this
//      / higher to make it less restrictive, usually at the cost of more execution time.
//      /
//      / Increasing this will make it more likely to find a good result.
//      /
//      / Decreasing this will speed up execution.
//      /
//      / This ratio is not exposed as a parameter in the original paper, but is instead computed
//      / recursively for a few iterations. It is roughly equivalent to the **reciprocal** of the
//      / **probability of rejecting a good model**. You can use that to control the probability
//      / that a good model is rejected.
//      /
//      / Default: `1e3`
    double likelihood_ratio_threshold_ = 1e1;
    // estimated during runtime
    double pos_likelihood_ratio_ = 0.0;
    double neg_likelihood_ratio_ = 0.0;
    size_t sprt_starting_iter_ = 52;
  };

  struct RansacStatistics
  {
    uint32_t num_iterations;
    int best_num_inliers;
    double best_model_score;
    double inlier_ratio;
    double delta;
    std::vector<int> inlier_indices;
    int number_lo_iterations;
  };

  class RansacBase
  {
  protected:
    void ResetStatistics(RansacStatistics *statistics) const
    {
      RansacStatistics &stats = *statistics;
      stats.best_num_inliers = 0;
      stats.best_model_score = std::numeric_limits<double>::max();
      stats.num_iterations = 0u;
      stats.inlier_ratio = 0.0;
      stats.delta = 0.0;
      stats.inlier_indices.clear();
      stats.number_lo_iterations = 0;
    }
  };

  // Implements LO-RANSAC with MSAC (top-hat) scoring, based on the description
  // provided in [Lebeda, Matas, Chum, Fixing the Locally Optimized RANSAC, BMVC
  // 2012]. Iteratively re-weighted least-squares optimization is optional.
  template <class Model, class ModelVector, class Solver,
            class Sampler = UniformSampling<Solver>>
  class LocallyOptimizedMSAC : public RansacBase
  {
  public:
    // Estimates a model using a given solver. Notice that the solver contains
    // all data and is responsible to implement a non-minimal solver and
    // least-squares refinement. The latter two are optional, i.e., a dummy
    // implementation returning false is sufficient.
    // Returns the number of inliers.
    int EstimateModel(LORansacOptions &options, const Solver &solver,
                      Model *best_model, RansacStatistics *statistics) const
    {
      ResetStatistics(statistics);
      RansacStatistics &stats = *statistics;

#ifdef DEBUG
      std::cout << "[" << solver.name() << "]" << std::endl;
#endif

      // Sanity check: No need to run RANSAC if there are not enough data
      // points.
      const int kMinSampleSize = solver.min_sample_size();
#ifdef DEBUG
      std::cout << "[" << solver.name() << "]" << "[" << kMinSampleSize << "]: " << std::endl;
#endif
      const int kNumData = solver.num_data();
      if (kMinSampleSize > kNumData || kMinSampleSize <= 0)
      {
        assert(false && "Minimum sample size is invalid");
        return 0;
      }

      // Initializes variables, etc.
      Sampler sampler(options.random_seed_, solver);
      std::mt19937 rng;
      rng.seed(options.random_seed_);

      uint32_t max_num_iterations =
          std::max(options.max_num_iterations_, options.min_num_iterations_);

      const double kSqrInlierThresh = options.squared_inlier_threshold_;
#ifdef DEBUG
      std::cout << "[" << solver.name() << "]" << "[threshold: " << kSqrInlierThresh << "]: " << std::endl;
#endif

      Model best_minimal_model;
      double best_min_model_score = std::numeric_limits<double>::max();

      std::vector<int> minimal_sample(kMinSampleSize);
      ModelVector estimated_models;

      // Runs random sampling.
      for (stats.num_iterations = 0u; stats.num_iterations < max_num_iterations;
           ++stats.num_iterations)
      {
        // As proposed by Lebeda et al., Local Optimization is not executed in
        // the first lo_starting_iterations_ iterations. We thus run LO on the
        // best model found so far once we reach this iteration.
        if (stats.num_iterations == options.lo_starting_iterations_ &&
            best_min_model_score < std::numeric_limits<double>::max())
        {
          ++stats.number_lo_iterations;
          LocalOptimization(options, solver, &rng, best_model,
                            &(stats.best_model_score));
#ifdef DEBUG
          std::cout << "[" << solver.name() << "]" << "[lo_starting_iterations]: " << stats.best_model_score << std::endl;
#endif
          // Updates the number of RANSAC iterations.
          auto cur_inliers = GetInliers(
              solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
          // std::cout << cur_inliers << ", " << stats.best_num_inliers  << std::endl;
          // assert(cur_inliers >= stats.best_num_inliers && "cost is better though we got less inliers");
          stats.best_num_inliers = cur_inliers;
          
          stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                               static_cast<double>(kNumData);
          max_num_iterations = utils::NumRequiredIterations(
              stats.inlier_ratio, 1.0 - options.success_probability_,
              kMinSampleSize, options.min_num_iterations_,
              options.max_num_iterations_);
        }

        sampler.Sample(&minimal_sample);

        // MinimalSolver returns the number of estimated models.
        const int kNumEstimatedModels =
            solver.MinimalSolver(minimal_sample, &estimated_models);

        if (kNumEstimatedModels <= 0)
          continue;

        // Finds the best model among all estimated models.
        double best_local_score = std::numeric_limits<double>::max();
        int best_local_model_id = 0;

        double worst_local_score = std::numeric_limits<double>::min();
        int worst_local_model_id = 0;

        GetBestEstimatedModelId(options, solver, estimated_models, kNumEstimatedModels,
                                kSqrInlierThresh, &best_local_score,
                                &best_local_model_id, &worst_local_score, &worst_local_model_id);

        // estimate delta - number of points consistent with any model
        // inspired by the rust implementation of ARRSAC
        // https://github.com/rust-cv/arrsac/blob/main/src/lib.rs
        // where delta estimation is computed as number of inliers of
        // worst model
        if (stats.num_iterations < options.sprt_starting_iter_) {
            auto delta = static_cast<double>(GetInliers(
                    solver, estimated_models[worst_local_model_id],
                    options.squared_inlier_threshold_, nullptr
            ));

            delta = std::max(delta, static_cast<double>(solver.min_sample_size()));
            delta = delta / static_cast<double> (solver.num_data());

            stats.delta = stats.delta == 0 ? delta : std::min(stats.delta, delta);
        } else if (stats.num_iterations == options.sprt_starting_iter_) {
            assert(stats.delta < stats.inlier_ratio && "delta should be less than eps if we don't want to reject better model");
            options.pos_likelihood_ratio_ = stats.delta / stats.inlier_ratio;
            options.neg_likelihood_ratio_ = (1.0 - stats.delta) / (1.0 - stats.inlier_ratio);
        }

#ifdef DEBUG
        std::cout << "[" << solver.name() << "]" << "[" << stats.num_iterations << "]" << "[local score]: " << best_local_score << std::endl;
#endif
        // Updates the best model found so far.
        if (best_local_score < best_min_model_score ||
            stats.num_iterations == options.lo_starting_iterations_)
        {
          const bool kBestMinModel = best_local_score < best_min_model_score;

          if (kBestMinModel)
          {
            // New best model (estimated from inliers found. Stores this model
            // and runs local optimization.
            best_min_model_score = best_local_score;
            best_minimal_model = estimated_models[best_local_model_id];

            // Updates the best model.
            UpdateBestModel(best_min_model_score, best_minimal_model,
                            &(stats.best_model_score), best_model);
          }

          const bool kRunLO =
              (stats.num_iterations >= options.lo_starting_iterations_ &&
               best_min_model_score < std::numeric_limits<double>::max());

          if ((!kBestMinModel) && (!kRunLO)) {
            assert("false" && "we are in update step though either of possibilities are not satisfied");
            continue;
          }

          // Performs local optimization. By construction, the local optimization
          // method returns the best model between all models found by local
          // optimization and the input model, i.e., score_refined_model <=
          // best_min_model_score holds.
          if (kRunLO)
          {
            ++stats.number_lo_iterations;
            double score = best_min_model_score;
            LocalOptimization(options, solver, &rng, &best_minimal_model, &score);
#ifdef DEBUG
            std::cout << "[" << solver.name() << "]" << "[local optimization]: " << score << std::endl;
#endif
            // Updates the best model.
            UpdateBestModel(score, best_minimal_model, &(stats.best_model_score),
                            best_model);
          }

          // Updates the number of RANSAC iterations.
          auto cur_inliers = GetInliers(
              solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
          // std::cout << cur_inliers << ", " << stats.best_num_inliers  << std::endl;
          // assert(cur_inliers >= stats.best_num_inliers && "cost is better though we got less inliers");
          stats.best_num_inliers = cur_inliers;
          
          stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                               static_cast<double>(kNumData);
          max_num_iterations = utils::NumRequiredIterations(
              stats.inlier_ratio, 1.0 - options.success_probability_,
              kMinSampleSize, options.min_num_iterations_,
              options.max_num_iterations_);
        }
      }

      // As proposed by Lebeda et al., Local Optimization is not executed in
      // the first lo_starting_iterations_ iterations. If LO-MSAC needs less than
      // lo_starting_iterations_ iterations, we run LO now.
      if (stats.num_iterations <= options.lo_starting_iterations_ &&
          stats.best_model_score < std::numeric_limits<double>::max())
      {
        ++stats.number_lo_iterations;
        LocalOptimization(options, solver, &rng, best_model,
                          &(stats.best_model_score));
#ifdef DEBUG
        std::cout << "[" << solver.name() << "]" << "[less than lo_starting_iteration finish]: " << stats.best_model_score << std::endl;
#endif
        auto cur_inliers = GetInliers(
              solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
          // assert(cur_inliers >= stats.best_num_inliers && "cost is better though we got less inliers");
          stats.best_num_inliers = cur_inliers;
                   
        stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                             static_cast<double>(kNumData);
      }

      if (options.final_least_squares_)
      {
        Model refined_model = *best_model;
        solver.LeastSquares(stats.inlier_indices, &refined_model);

        double score = std::numeric_limits<double>::max();
        ScoreModel(options, solver, refined_model, kSqrInlierThresh, &score);
        if (score < stats.best_model_score)
        {
          stats.best_model_score = score;
#ifdef DEBUG
          std::cout << "[" << solver.name() << "]" << "[final least squares]: " << score << std::endl;
#endif
          *best_model = refined_model;

          auto cur_inliers = GetInliers(
              solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
          // assert(cur_inliers >= stats.best_num_inliers && "cost is better though we got less inliers");
          stats.best_num_inliers = cur_inliers;

          stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                               static_cast<double>(kNumData);
        }
      }

      return stats.best_num_inliers;
    }

  protected:
    void GetBestEstimatedModelId(const LORansacOptions& options, const Solver &solver, const ModelVector &models,
                                 const int num_models,
                                 const double squared_inlier_threshold,
                                 double *best_score, int *best_model_id,
                                 double *worst_score, int *worst_model_id) const
    {
      *best_score = std::numeric_limits<double>::max();
      *best_model_id = 0;


      for (int m = 0; m < num_models; ++m)
      {
        double score = std::numeric_limits<double>::max();
        ScoreModel(options, solver, models[m], squared_inlier_threshold, &score);

        if (score < *best_score)
        {
          *best_score = score;
          *best_model_id = m;
        }

        if (score > *worst_score) {
            *worst_score = score;
            *worst_model_id = m;
        }
      }
    }

    void ScoreModel(const LORansacOptions& options, const Solver &solver, const Model &model,
                    const double squared_inlier_threshold, double *score, bool sprt = false) const
    {
      double likelihood = 1.0;

      const int kNumData = solver.num_data();
      *score = 0.0;
      for (int i = 0; i < kNumData; ++i)
      {
        double squared_error = solver.EvaluateModelOnPoint(model, i);

        // if we estimated initial delta and epsilon
        if (options.pos_likelihood_ratio_ && options.neg_likelihood_ratio_) {
            if (squared_error < options.squared_inlier_threshold_) {
                likelihood *= options.pos_likelihood_ratio_;
            } else {
                likelihood *= options.neg_likelihood_ratio_;
            }
//            std::cout << "lik: "<< likelihood << " pos: " << options.pos_likelihood_ratio_ << " neg: " << options.neg_likelihood_ratio_ << std::endl;

            if (likelihood > options.likelihood_ratio_threshold_  || likelihood == std::numeric_limits<double>::max()) {
                std::cout << "skipped" << std::endl;
                *score = std::numeric_limits<double>::max();
                return;
            }
        }

        *score += ComputeScore(squared_error, squared_inlier_threshold);
      }
    }

    // MSAC (top-hat) scoring function.
    inline double ComputeScore(const double squared_error,
                               const double squared_error_threshold) const
    {
      return std::min(squared_error, squared_error_threshold);
    }

    int GetInliers(const Solver &solver, const Model &model,
                   const double squared_inlier_threshold,
                   std::vector<int> *inliers) const
    {
      const int kNumData = solver.num_data();
      if (inliers == nullptr)
      {
        int num_inliers = 0;
        for (int i = 0; i < kNumData; ++i)
        {
          double squared_error = solver.EvaluateModelOnPoint(model, i);
          if (squared_error < squared_inlier_threshold)
          {
            ++num_inliers;
          }
        }
        return num_inliers;
      }
      else
      {
        inliers->clear();
        int num_inliers = 0;
        for (int i = 0; i < kNumData; ++i)
        {
          double squared_error = solver.EvaluateModelOnPoint(model, i);
          if (squared_error < squared_inlier_threshold)
          {
            ++num_inliers;
            inliers->push_back(i);
          }
        }
        return num_inliers;
      }
    }

    // See algorithms 2 and 3 in Lebeda et al.
    // The input model is overwritten with the refined model if the latter is
    // better, i.e., has a lower score.
    void LocalOptimization(const LORansacOptions &options, const Solver &solver,
                           std::mt19937 *rng, Model *best_minimal_model,
                           double *score_best_minimal_model) const
    {
      const int kNumData = solver.num_data();
      // kMinNonMinSampleSize stores how many data points are required for a
      // non-minimal sample. For example, consider the case of pose estimation
      // for a calibrated camera. A minimal sample has size 3, while the
      // smallest non-minimal sample has size 4.
      const int kMinNonMinSampleSize = solver.non_minimal_sample_size();
      if (kMinNonMinSampleSize > kNumData) {
        assert(false && "Min amount of data needed for LO is less than data provided");
        return;
      }

      const int kMinSampleSize = solver.min_sample_size();

      const double kSqInThresh = options.squared_inlier_threshold_;
      const double kThreshMult = options.threshold_multiplier_;

      // Performs an initial least squares fit of the best model found by the
      // minimal solver so far and then determines the inliers to that model
      // under a (slightly) relaxed inlier threshold.
      Model m_init = *best_minimal_model;
      LeastSquaresFit(options, kSqInThresh * kThreshMult, solver, rng, &m_init);

      double score = std::numeric_limits<double>::max();
      ScoreModel(options, solver, m_init, kSqInThresh, &score);
      UpdateBestModel(score, m_init, score_best_minimal_model,
                      best_minimal_model);

      std::vector<int> inliers_base;
      GetInliers(solver, m_init, kSqInThresh * kThreshMult, &inliers_base);

      // Determines the size of the non-miminal samples drawn in each LO step.
      const int kNonMinSampleSize =
          std::max(kMinNonMinSampleSize,
                   std::min(kMinSampleSize * options.non_min_sample_multiplier_,
                            static_cast<int>(inliers_base.size()) / 2));

      // Performs the actual local optimization (LO).
      std::vector<int> sample;
      for (int r = 0; r < options.num_lo_steps_; ++r)
      {
        sample = inliers_base;
        utils::RandomShuffleAndResize(kNonMinSampleSize, rng, &sample);

        Model m_non_min;
        if (!solver.NonMinimalSolver(sample, &m_non_min))
          continue;

        ScoreModel(options, solver, m_non_min, kSqInThresh, &score);
        UpdateBestModel(score, m_non_min, score_best_minimal_model,
                        best_minimal_model);

        // Iterative least squares refinement.
        LeastSquaresFit(options, kSqInThresh, solver, rng, &m_non_min);

        // The current threshold multiplier and its update.
        double thresh = kThreshMult * kSqInThresh;
        double thresh_mult_update =
            (kThreshMult - 1.0) * kSqInThresh /
            static_cast<int>(options.num_lsq_iterations_ - 1);
        for (int i = 0; i < options.num_lsq_iterations_; ++i)
        {
          LeastSquaresFit(options, thresh, solver, rng, &m_non_min);

          ScoreModel(options, solver, m_non_min, kSqInThresh, &score);
          UpdateBestModel(score, m_non_min, score_best_minimal_model,
                          best_minimal_model);
          thresh -= thresh_mult_update;
        }
      }
    }

    void LeastSquaresFit(const LORansacOptions &options, const double thresh,
                         const Solver &solver, std::mt19937 *rng,
                         Model *model) const
    {
      const int kLSqSampleSize =
          options.min_sample_multiplicator_ * solver.min_sample_size();
      std::vector<int> inliers;
      int num_inliers = GetInliers(solver, *model, thresh, &inliers);
      if (num_inliers < solver.min_sample_size())
        return;
      int lsq_data_size = std::min(kLSqSampleSize, num_inliers);
      utils::RandomShuffleAndResize(lsq_data_size, rng, &inliers);
      solver.LeastSquares(inliers, model);
    }

    inline void UpdateBestModel(const double score_curr, const Model &m_curr,
                                double *score_best, Model *m_best) const
    {
      if (score_curr < *score_best)
      {
        *score_best = score_curr;
        *m_best = m_curr;
      }
    }
  };

  template <class Model, class ModelVector, class Solver, class FullSolver,
            class Sampler = UniformSampling<Solver>>
  class LocallyOptimizedTwoSolverMSAC : public RansacBase
  {
  public:
    // Estimates a model using a given solver. Notice that the solver contains
    // all data and is responsible to implement a non-minimal solver and
    // least-squares refinement. The latter two are optional, i.e., a dummy
    // implementation returning false is sufficient.
    // Returns the number of inliers.
    int EstimateModel(const LORansacOptions &options, const Solver &solver,
                      const FullSolver &fullSolver, Model *best_model,
                      RansacStatistics *statistics) const
    {
      ResetStatistics(statistics);
      RansacStatistics &stats = *statistics;

      // Sanity check: No need to run RANSAC if there are not enough data
      // points.
      const int kMinSampleSize = solver.min_sample_size();
      const int kNumData = solver.num_data();
      if (kMinSampleSize > kNumData || kMinSampleSize <= 0)
      {
        return 0;
      }

      // Initializes variables, etc.
      Sampler sampler(options.random_seed_, solver);
      std::mt19937 rng;
      rng.seed(options.random_seed_);

      uint32_t max_num_iterations =
          std::max(options.max_num_iterations_, options.min_num_iterations_);

      const double kSqrInlierThresh = options.squared_inlier_threshold_;

      Model best_minimal_model;
      double best_min_model_score = std::numeric_limits<double>::max();

      std::vector<int> minimal_sample(kMinSampleSize);
      ModelVector estimated_models;

      // Runs random sampling.
      for (stats.num_iterations = 0u; stats.num_iterations < max_num_iterations;
           ++stats.num_iterations)
      {
        // As proposed by Lebeda et al., Local Optimization is not executed in
        // the first lo_starting_iterations_ iterations. We thus run LO on the
        // best model found so far once we reach this iteration.
        if (stats.num_iterations == options.lo_starting_iterations_ &&
            best_min_model_score < std::numeric_limits<double>::max())
        {
          ++stats.number_lo_iterations;
          LocalOptimization(options, fullSolver, &rng, best_model,
                            &(stats.best_model_score));

          // Updates the number of RANSAC iterations.
          stats.best_num_inliers = GetInliers(
              solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
          stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                               static_cast<double>(kNumData);
          max_num_iterations = utils::NumRequiredIterations(
              stats.inlier_ratio, 1.0 - options.success_probability_,
              solver.min_sample_size(), options.min_num_iterations_,
              options.max_num_iterations_);
        }

        sampler.Sample(&minimal_sample);

        // MinimalSolver returns the number of estimated models.
        const int kNumEstimatedModels =
            solver.MinimalSolver(minimal_sample, &estimated_models);
        if (kNumEstimatedModels <= 0)
          continue;

        // Finds the best model among all estimated models.
        double best_local_score = std::numeric_limits<double>::max();
        int best_local_model_id = 0;
        GetBestEstimatedModelId(solver, estimated_models, kNumEstimatedModels,
                                kSqrInlierThresh, &best_local_score,
                                &best_local_model_id);

        //                std::cout << "Best local score:" << best_local_score << std::endl;

        // Updates the best model found so far.
        if (best_local_score < best_min_model_score ||
            stats.num_iterations == options.lo_starting_iterations_)
        {
          const bool kBestMinModel = best_local_score < best_min_model_score;

          if (kBestMinModel)
          {
            // New best model (estimated from inliers found. Stores this model
            // and runs local optimization.
            best_min_model_score = best_local_score;
            best_minimal_model = estimated_models[best_local_model_id];

            // Updates the best model.
            UpdateBestModel(best_min_model_score, best_minimal_model,
                            &(stats.best_model_score), best_model);
          }

          const bool kRunLO =
              (stats.num_iterations >= options.lo_starting_iterations_ &&
               best_min_model_score < std::numeric_limits<double>::max());

          if ((!kBestMinModel) && (!kRunLO))
            continue;

          // Performs local optimization. By construction, the local optimization
          // method returns the best model between all models found by local
          // optimization and the input model, i.e., score_refined_model <=
          // best_min_model_score holds.
          if (kRunLO)
          {
            ++stats.number_lo_iterations;
            double score = best_min_model_score;
            LocalOptimization(options, fullSolver, &rng, &best_minimal_model, &score);

            //                        std::cout << "After LO 2: " << score << std::endl;

            // Updates the best model.
            UpdateBestModel(score, best_minimal_model, &(stats.best_model_score),
                            best_model);
          }

          stats.best_num_inliers = GetInliers(
                solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));

          // Updates the number of RANSAC iterations.
          stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                               static_cast<double>(kNumData);
          max_num_iterations = utils::NumRequiredIterations(
              stats.inlier_ratio, 1.0 - options.success_probability_,
              solver.min_sample_size(), options.min_num_iterations_,
              options.max_num_iterations_);
        }
      }

      // As proposed by Lebeda et al., Local Optimization is not executed in
      // the first lo_starting_iterations_ iterations. If LO-MSAC needs less than
      // lo_starting_iterations_ iterations, we run LO now.
      if (stats.num_iterations <= options.lo_starting_iterations_ &&
          stats.best_model_score < std::numeric_limits<double>::max())
      {
        ++stats.number_lo_iterations;
        LocalOptimization(options, fullSolver, &rng, best_model,
                          &(stats.best_model_score));

        //                std::cout << "After LO: " << stats.best_model_score << std::endl;

        stats.best_num_inliers = GetInliers(solver, *best_model, kSqrInlierThresh,
                                            &(stats.inlier_indices));
        stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                             static_cast<double>(kNumData);
      }

      if (options.final_least_squares_)
      {
        Model refined_model = *best_model;
        fullSolver.LeastSquares(stats.inlier_indices, &refined_model);

        double score = std::numeric_limits<double>::max();
        ScoreModel(solver, refined_model, kSqrInlierThresh, &score);
        if (score < stats.best_model_score)
        {
          stats.best_model_score = score;

          // std::cout << "Final score: " << score << std::endl;

          *best_model = refined_model;

          stats.best_num_inliers = GetInliers(
              solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
          stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                               static_cast<double>(kNumData);
        }
      }

      return stats.best_num_inliers;
    }

  protected:
    template <typename Solv>
    void GetBestEstimatedModelId(const Solv &solver, const ModelVector &models,
                                 const int num_models,
                                 const double squared_inlier_threshold,
                                 double *best_score, int *best_model_id) const
    {
      *best_score = std::numeric_limits<double>::max();
      *best_model_id = 0;
      for (int m = 0; m < num_models; ++m)
      {
        double score = std::numeric_limits<double>::max();
        ScoreModel(solver, models[m], squared_inlier_threshold, &score);

        if (score < *best_score)
        {
          *best_score = score;
          *best_model_id = m;
        }
      }
    }

    template <typename Solv>
    void ScoreModel(const Solv &solver, const Model &model,
                    const double squared_inlier_threshold, double *score) const
    {
      const int kNumData = solver.num_data();
      *score = 0.0;
      for (int i = 0; i < kNumData; ++i)
      {
        double squared_error = solver.EvaluateModelOnPoint(model, i);
        *score += ComputeScore(squared_error, squared_inlier_threshold);
      }
    }

    // MSAC (top-hat) scoring function.
    inline double ComputeScore(const double squared_error,
                               const double squared_error_threshold) const
    {
      return std::min(squared_error, squared_error_threshold);
    }

    template <typename Solv>
    int GetInliers(const Solv &solver, const Model &model,
                   const double squared_inlier_threshold,
                   std::vector<int> *inliers) const
    {
      const int kNumData = solver.num_data();
      if (inliers == nullptr)
      {
        int num_inliers = 0;
        for (int i = 0; i < kNumData; ++i)
        {
          double squared_error = solver.EvaluateModelOnPoint(model, i);
          if (squared_error < squared_inlier_threshold)
          {
            ++num_inliers;
          }
        }
        return num_inliers;
      }
      else
      {
        inliers->clear();
        int num_inliers = 0;
        for (int i = 0; i < kNumData; ++i)
        {
          double squared_error = solver.EvaluateModelOnPoint(model, i);
          if (squared_error < squared_inlier_threshold)
          {
            ++num_inliers;
            inliers->push_back(i);
          }
        }
        return num_inliers;
      }
    }

    // See algorithms 2 and 3 in Lebeda et al.
    // The input model is overwritten with the refined model if the latter is
    // better, i.e., has a lower score.
    template <typename Solv>
    void LocalOptimization(const LORansacOptions &options, const Solv &solver,
                           std::mt19937 *rng, Model *best_minimal_model,
                           double *score_best_minimal_model) const
    {
      const int kNumData = solver.num_data();
      // kMinNonMinSampleSize stores how many data points are required for a
      // non-minimal sample. For example, consider the case of pose estimation
      // for a calibrated camera. A minimal sample has size 3, while the
      // smallest non-minimal sample has size 4.
      const int kMinNonMinSampleSize = solver.non_minimal_sample_size();
      if (kMinNonMinSampleSize > kNumData)
        return;

      const int kMinSampleSize = solver.min_sample_size();

      const double kSqInThresh = options.squared_inlier_threshold_;
      const double kThreshMult = options.threshold_multiplier_;

      // Performs an initial least squares fit of the best model found by the
      // minimal solver so far and then determines the inliers to that model
      // under a (slightly) relaxed inlier threshold.
      Model m_init = *best_minimal_model;
      LeastSquaresFit(options, kSqInThresh * kThreshMult, solver, rng, &m_init);

      double score = std::numeric_limits<double>::max();
      ScoreModel(solver, m_init, kSqInThresh, &score);
      UpdateBestModel(score, m_init, score_best_minimal_model,
                      best_minimal_model);

      std::vector<int> inliers_base;
      GetInliers(solver, m_init, kSqInThresh * kThreshMult, &inliers_base);

      // Determines the size of the non-miminal samples drawn in each LO step.
      const int kNonMinSampleSize =
          std::max(kMinNonMinSampleSize,
                   std::min(kMinSampleSize * options.non_min_sample_multiplier_,
                            static_cast<int>(inliers_base.size()) / 2));

      // Performs the actual local optimization (LO).
      std::vector<int> sample;
      for (int r = 0; r < options.num_lo_steps_; ++r)
      {
        sample = inliers_base;
        utils::RandomShuffleAndResize(kNonMinSampleSize, rng, &sample);

        Model m_non_min;
        if (!solver.NonMinimalSolver(sample, &m_non_min))
          continue;

        ScoreModel(solver, m_non_min, kSqInThresh, &score);
        UpdateBestModel(score, m_non_min, score_best_minimal_model,
                        best_minimal_model);

        // Iterative least squares refinement.
        LeastSquaresFit(options, kSqInThresh, solver, rng, &m_non_min);

        // The current threshold multiplier and its update.
        double thresh = kThreshMult * kSqInThresh;
        double thresh_mult_update =
            (kThreshMult - 1.0) * kSqInThresh /
            static_cast<int>(options.num_lsq_iterations_ - 1);
        for (int i = 0; i < options.num_lsq_iterations_; ++i)
        {
          LeastSquaresFit(options, thresh, solver, rng, &m_non_min);

          ScoreModel(solver, m_non_min, kSqInThresh, &score);
          UpdateBestModel(score, m_non_min, score_best_minimal_model,
                          best_minimal_model);
          thresh -= thresh_mult_update;
        }
      }
    }

    template <typename Solv>
    void LeastSquaresFit(const LORansacOptions &options, const double thresh,
                         const Solv &solver, std::mt19937 *rng,
                         Model *model) const
    {
      const int kLSqSampleSize =
          options.min_sample_multiplicator_ * solver.min_sample_size();
      std::vector<int> inliers;
      int num_inliers = GetInliers(solver, *model, thresh, &inliers);
      if (num_inliers < solver.min_sample_size())
        return;
      int lsq_data_size = std::min(kLSqSampleSize, num_inliers);
      utils::RandomShuffleAndResize(lsq_data_size, rng, &inliers);
      solver.LeastSquares(inliers, model);
    }

    inline void UpdateBestModel(const double score_curr, const Model &m_curr,
                                double *score_best, Model *m_best) const
    {
      if (score_curr < *score_best)
      {
        *score_best = score_curr;
        *m_best = m_curr;
      }
    }
  };

} // namespace ransac_lib

#endif // RANSACLIB_RANSACLIB_RANSAC_H_
