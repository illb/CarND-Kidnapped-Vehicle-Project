/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  num_particles = 100;
  
  // create normal (Gaussian) distributions for x, y, theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  std::default_random_engine gen;
  
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  
  if (fabs(yaw_rate) < 0.0000001) {
    if (yaw_rate > 0.0) {
      yaw_rate = 0.0000001;
    } else {
      yaw_rate = -0.0000001;
    }
  }
  
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  default_random_engine gen;
  
  for (int i = 0; i < num_particles; ++i) {
    Particle& p = particles[i];
    
    auto x = p.x;
    auto y = p.y;
    auto theta = p.theta;
    
    x = x + (velocity / yaw_rate) * (sin(theta + (yaw_rate * delta_t)) - sin(theta));
    y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + (yaw_rate * delta_t)));
    theta = theta + (yaw_rate * delta_t);
    
    p.x = x + dist_x(gen);
    p.y = y + dist_y(gen);
    p.theta = theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  
  for (int i = 0; i < observations.size(); i++) {
    auto& o = observations[i];
    double min_distance = 1000000.0;
    for (int j = 0; j < predicted.size(); j++) {
      auto& p = predicted[j];
      double distance = dist(o.x, o.y, p.x, p.y);
      if (distance < min_distance) {
        min_distance = distance;
        observations[i].id = p.id;
      }
    }
  }
}


inline double bivariate_gaussian(double x, double y, double mu_x, double mu_y, double std_x, double std_y) {
  double r = exp((-1.0 / 2.0)
                 * (pow(x - mu_x, 2) / pow(std_x, 2) + pow(y - mu_y, 2) / pow(std_y, 2))
                 )
             / (2.0 * M_PI * std_x * std_y);
  return r;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  
  for (int i = 0; i < num_particles; ++i) {
    auto& particle = particles[i];
    
    auto p_x = particle.x;
    auto p_y = particle.y;
    auto p_theta = particle.theta;
    
    vector<LandmarkObs> t_observations;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs landmark;
      LandmarkObs& o = observations[j];
      landmark.id = 0;
      landmark.x = o.x * cos(p_theta) - o.y * sin(p_theta) + p_x;
      landmark.y = o.x * sin(p_theta) + o.y * cos(p_theta) + p_y;
      t_observations.push_back(landmark);
    }

    vector<LandmarkObs> predictions;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      auto& current = map_landmarks.landmark_list[j];
      auto distance = dist(p_x, p_y, current.x_f, current.y_f);
      if (distance <= sensor_range) {
        LandmarkObs landmark;
        landmark.id = current.id_i;
        landmark.x = current.x_f;
        landmark.y = current.y_f;
        predictions.push_back(landmark);
      }
    }
    
    // find nearest neighbor
    dataAssociation(predictions, t_observations);
    
    double weight = 1.0;
    for (int j = 0; j < t_observations.size(); j++) {
      auto& o = t_observations[j];
      
      for (int k = 0; k < predictions.size(); k++) {
        auto& p = predictions[k];
        if (o.id == p.id) {
          weight *= bivariate_gaussian(o.x, o.y, p.x, p.y, std_landmark[0], std_landmark[1]);
        }
      }
    }
    
    particle.weight = weight;
    weights[i] = particle.weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  discrete_distribution<> d(weights.begin(), weights.end());
  
  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; ++i) {
    new_particles.push_back(particles[d(gen)]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
