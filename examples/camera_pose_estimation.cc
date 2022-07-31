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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <algorithm>
#include <set>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <RansacLib/ransac.h>
#include "calibrated_absolute_pose_estimator.h"

namespace ransac_lib
{
  namespace calibrated_absolute_pose
  {

    std::vector<Eigen::Matrix3d> add_some_noise(double deviation)
    {
      Eigen::Matrix3d Rz, Rx;

      std::random_device rd;                                           // obtain a random number from hardware
      std::mt19937 gen(rd());                                          // seed the generator
      std::uniform_real_distribution<> distr(-deviation / 2, deviation / 2); // define the range
      double x = distr(gen);
      double z = deviation - std::abs(x);
      z = distr(gen) > 0 ? z : -z;
      assert(std::abs(x) + std::abs(z) == deviation);

      double cx = std::cos(x * 3.14159 / 180);
      double sx = -1 * std::sin(x * 3.14159 / 180);

      double cz = std::cos(z * 3.14159 / 180);
      double sz = -1 * std::sin(z * 3.14159 / 180);

      Rz << cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0;
      Rx << 1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx;

      return std::vector<Eigen::Matrix3d>{Rx, Rz};
    }

    void GenerateRandomInstance(const double width, const double height,
                                const double focal_length, const int num_inliers,
                                const int num_outliers, double inlier_threshold,
                                const double min_depth, const double max_depth,
                                const size_t deviation, Points2D *points2D,
                                ViewingRays *rays, Points3D *points3D)
    {
      const int kNumPoints = num_inliers + num_outliers;
      points2D->resize(kNumPoints);
      points3D->resize(kNumPoints);

      std::vector<int> indices(kNumPoints);
      std::iota(indices.begin(), indices.end(), 0);

      std::random_device rand_dev;
      std::mt19937 rng(rand_dev());

      std::shuffle(indices.begin(), indices.end(), rng);

      const double kWidthHalf = width * 0.5;
      const double kHeightHalf = height * 0.5;
      std::uniform_real_distribution<double> distr_x(-kWidthHalf, kWidthHalf);
      std::uniform_real_distribution<double> distr_y(-kHeightHalf, kHeightHalf);
      std::uniform_real_distribution<double> distr_d(min_depth, max_depth);
      std::uniform_real_distribution<double> distr(-1.0, 1.0);

      // Generates the inliers.
      for (int i = 0; i < num_inliers; ++i)
      {
        const int kIndex = indices[i];
        (*points2D)[kIndex] = Eigen::Vector2d(distr_x(rng), distr_y(rng));

        Eigen::Vector3d dir = (*points2D)[kIndex].homogeneous();
        dir.head<2>() /= focal_length;
        dir.normalize();

        // Obtains the 3D point.
        (*points3D)[kIndex] = dir * distr_d(rng);

        // Adds some noise to the 2D position to make the case more realistic.
        (*points2D)[kIndex] +=
            Eigen::Vector2d(distr(rng), distr(rng)) * inlier_threshold;
      }

      // Generates the outlier.
      for (int i = num_inliers; i < kNumPoints; ++i)
      {
        const int kIndex = indices[i];
        Eigen::Vector2d p(distr_x(rng), distr_y(rng));

        Eigen::Vector3d dir = p.homogeneous();
        dir.head<2>() /= focal_length;
        dir.normalize();

        // Obtains the 3D point.
        (*points3D)[kIndex] = dir * distr_d(rng);

        // Estimates a new pixel position that is far enough from the original one.
        (*points2D)[kIndex] = Eigen::Vector2d(distr_x(rng), distr_y(rng));
        while (((*points2D)[kIndex] - p).norm() < 10.0 * (inlier_threshold + 1.0))
        {
          (*points2D)[kIndex] = Eigen::Vector2d(distr_x(rng), distr_y(rng));
        }
      }

      Eigen::Vector2d r;
      r.setRandom().normalize();

      Eigen::Matrix3d R;
      R << r(0), 0.0, r(1), 0.0, 1.0, 0.0, -r(1), 0.0, r(0);
      auto noise = add_some_noise(deviation);
      auto Rx = std::move(noise[0]);
      auto Rz = std::move(noise[1]);

      R = Rz * Rx * R;

      //  // Randomly rotates and translates the 3D points.
      //  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
      //  Eigen::Matrix3d R(q);

      std::uniform_real_distribution<double> distr_scale(1.0, 2.0);
      Eigen::Vector3d t(distr(rng), distr(rng), distr(rng));
      t *= distr_scale(rng);

      for (int i = 0; i < kNumPoints; ++i)
      {
        Eigen::Vector3d p = R * (*points3D)[i] + t;
        (*points3D)[i] = p;
      }

      CalibratedAbsolutePoseEstimator::PixelsToViewingRays(
          focal_length, focal_length, *points2D, rays);
    }

  } // namespace calibrated_absolute_pose
} // namespace ransac_lib

std::vector<std::string> split(const std::string& str, const std::string& regex_str)
{
    std::regex regexz(regex_str);
    std::vector<std::string> list(std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                  std::sregex_token_iterator());
    return list;
}

std::vector<double> config_to_vec(std::ifstream& f) {
  std::string line;

  f >> line;
  auto substrs = split(line, "=");
  auto params = std::move(substrs[1]);
  auto splited_params = split(params, ",");

  std::vector<double> res;
  res.reserve(splited_params.size());
  std::for_each(splited_params.begin(), splited_params.end(), 
    [&res](const auto& elm){ res.emplace_back(std::atof(elm.c_str())); });

  return res;
}


template<typename Ransac, typename Solver, typename OtherSolver>
void run_solver(Ransac ransac, Solver solver, OtherSolver full_solver, ransac_lib::LORansacOptions& options) {
  ransac_lib::RansacStatistics ransac_stats;
  ransac_lib::calibrated_absolute_pose::CameraPose best_model;

  auto ransac_start = std::chrono::system_clock::now();
  int num_inliers = ransac.EstimateModel(options, solver, full_solver, &best_model, &ransac_stats);
  auto ransac_end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = ransac_end - ransac_start;
  std::cout << num_inliers << ", " << ransac_stats.num_iterations << ", " << ransac_stats.inlier_ratio << ", "
            << ransac_stats.best_model_score << ", " << elapsed_seconds.count() << ", "
            << ransac_stats.number_lo_iterations << std::endl;
}

template<typename Ransac, typename Solver>
void run_solver(Ransac ransac, Solver solver, ransac_lib::LORansacOptions& options) {
  ransac_lib::RansacStatistics ransac_stats;
  ransac_lib::calibrated_absolute_pose::CameraPose best_model;

  auto ransac_start = std::chrono::system_clock::now();
  int num_inliers = ransac.EstimateModel(options, solver, &best_model, &ransac_stats);
  auto ransac_end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = ransac_end - ransac_start;
  std::cout << num_inliers << ", " << ransac_stats.num_iterations << ", " << ransac_stats.inlier_ratio << ", "
            << ransac_stats.best_model_score << ", " << elapsed_seconds.count() << ", "
            << ransac_stats.number_lo_iterations << std::endl;
}

int main(int argc, char **argv)
{


  using ransac_lib::calibrated_absolute_pose::Points3D;
  using ransac_lib::calibrated_absolute_pose::ViewingRays;

  if (argc != 3) {
    std::cerr << "wrong usage : specify deviation" << std::endl;
    exit(1);
  }

  const size_t deviation = atoi(argv[2]);

  std::ifstream f(argv[1]);

  auto solvers_vec = config_to_vec(f);
  auto outlier_ratios = config_to_vec(f);

  std::set<int> solvers_to_run{solvers_vec.begin(), solvers_vec.end()};

  std::random_device rand_dev;
  const int kNumDataPoints = 2000;
  const double kWidth = 640.0;
  const double kHeight = 320.0;
  const double kFocalLength = (kWidth * 0.5) / std::tan(60.0 * M_PI / 180.0);
  const double kInThreshPX = 12.0;

  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = 100u;
  options.max_num_iterations_ = 1000000u;
  options.random_seed_ = rand_dev();
  options.squared_inlier_threshold_ = kInThreshPX * kInThreshPX;
  options.min_sample_multiplicator_ = 7;
  options.num_lsq_iterations_ = 4;
  options.num_lo_steps_ = 10;
  options.final_least_squares_ = true;

  for (const double outlier_ratio : outlier_ratios)
  {
    // std::cout << "running for: " << 1 - outlier_ratio << std::endl << std::endl;
    int num_outliers =
        static_cast<int>(static_cast<double>(kNumDataPoints) * outlier_ratio);
    int num_inliers = kNumDataPoints - num_outliers;

    ransac_lib::calibrated_absolute_pose::Points2D points2D;
    ViewingRays rays;
    Points3D points3D;
    ransac_lib::calibrated_absolute_pose::GenerateRandomInstance(
        kWidth, kHeight, kFocalLength, num_inliers, num_outliers, 2.0, 2.0,
        10.0, deviation, &points2D, &rays, &points3D);


    const ransac_lib::calibrated_absolute_pose::CalibratedAbsolutePoseEstimator
        fullSolver(kFocalLength, kFocalLength, kInThreshPX * kInThreshPX, points2D,
                   rays, points3D);

    // currently inlier threshold does not make any influence
    const ransac_lib::calibrated_absolute_pose::CalibratedAbsolutePoseEstimator2p
        solver(kFocalLength, kFocalLength, kInThreshPX * kInThreshPX, points2D,
               rays, points3D);

    // currently inlier threshold does not make any influence
    const ransac_lib::calibrated_absolute_pose::CalibratedAbsolutePoseEstimator2p
        dyn_solver(kFocalLength, kFocalLength, std::pow(1.0 / (1.0 - outlier_ratio), 1.0/2.0) * kInThreshPX * kInThreshPX, points2D,
               rays, points3D);

    ransac_lib::LocallyOptimizedTwoSolverMSAC<
        ransac_lib::calibrated_absolute_pose::CameraPose,
        ransac_lib::calibrated_absolute_pose::CameraPoses,
        ransac_lib::calibrated_absolute_pose::CalibratedAbsolutePoseEstimator2p,
        ransac_lib::calibrated_absolute_pose::CalibratedAbsolutePoseEstimator>
        flomsac;

    ransac_lib::LocallyOptimizedMSAC<
        ransac_lib::calibrated_absolute_pose::CameraPose,
        ransac_lib::calibrated_absolute_pose::CameraPoses,
        ransac_lib::calibrated_absolute_pose::CalibratedAbsolutePoseEstimator>
        p3plomsac;

    ransac_lib::LocallyOptimizedMSAC<
        ransac_lib::calibrated_absolute_pose::CameraPose,
        ransac_lib::calibrated_absolute_pose::CameraPoses,
        ransac_lib::calibrated_absolute_pose::CalibratedAbsolutePoseEstimator2p>
        p2plomsac;
        
    ransac_lib::LocallyOptimizedMSAC<
        ransac_lib::calibrated_absolute_pose::CameraPose,
        ransac_lib::calibrated_absolute_pose::CameraPoses,
        ransac_lib::calibrated_absolute_pose::CalibratedAbsolutePoseEstimator2p>
        dyn_p2plomsac;
    
    if (solvers_to_run.find(1) != solvers_to_run.end())
      run_solver(flomsac, solver, fullSolver, options);

    if (solvers_to_run.find(2) != solvers_to_run.end())
      run_solver(p2plomsac, solver, options);
  
    if (solvers_to_run.find(3) != solvers_to_run.end())
      run_solver(p3plomsac, fullSolver, options);

    if (solvers_to_run.find(4) != solvers_to_run.end())
      run_solver(dyn_p2plomsac, dyn_solver, options);
  }

  return 0;
}
