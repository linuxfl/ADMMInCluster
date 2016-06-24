#pragma once
#include <string>
#include <atomic>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>

#include "util/Eigen/Dense"

namespace LR {
class LinearRegression {
	public:
			LinearRegression();
			~LinearRegression();
			void Start();
			void soft_threshold(Eigen::VectorXf src,Eigen::VectorXf &des,float t);
	private:
			std::atomic<int> thread_counter_;
			
			// petuum parameters
			int client_id_, num_clients_, num_worker_threads_;
			int table_staleness_;
			
			// evaluate parameters
			int num_epochs_;
			
			// timer
			boost::posix_time::ptime initT_;
			//data
			int row,feature;
			
			//ADMM parameter
			float rho;
			float lambda;
			float errorthreshold;
			
			//data file
			std::string data_dir;
			std::string output_dir;
		
};
}; // namespace LR
