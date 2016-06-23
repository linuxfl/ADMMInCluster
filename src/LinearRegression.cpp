#include <string>
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <mutex>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <io/general_fstream.hpp>
#include <stdio.h>

#include "util/context.hpp"
#include "LinearRegression.hpp"

namespace LR {

	// Constructor
	LinearRegression::LinearRegression(): thread_counter_(0) {
		// timer
		initT_ = boost::posix_time::microsec_clock::local_time();
		
		/* context */
		lda::Context & context = lda::Context::get_instance();
		
		// objective function parameters
		feature = context.get_int32("feature");
		row = context.get_int32("row");
		
		// petuum parameters
		client_id_ = context.get_int32("client_id");
		num_clients_ = context.get_int32("num_clients");
		num_worker_threads_ = context.get_int32("num_worker_threads");
		
		// optimization parameters
		num_epochs_ = context.get_int32("num_epochs");
		table_staleness_ = context.get_int32("table_staleness");
		
		//ADMM parameters
		rho = context.get_double("rho");
		lambda = context.get_double("lambda");
		errorthreshold = context.get_double("errorthreshold");
		data_dir = context.get_string("data_dir");
		output_dir = context.get_string("output_dir");
		
		LOG(INFO) <<"row: "<<row<< " feature: " << feature << " rho: "<<rho << " lambda: "
		<< lambda <<" errorthreshold: " << errorthreshold;
}

	void LinearRegression::soft_threshold(Eigen::VectorXf src,Eigen::VectorXf &des,float t){
		for(int i =0; i < feature;i++){
			if(src(i) > t)
				des(i) = src(i) - t;
			else if(src(i) <= t && src(i) >= -t)
				des(i) = 0.0;
			else
				des(i) = src(i) + t;
		}
	}
	
	// linearRegression
	void LinearRegression::Start() {
		// thread id on a client
		int thread_id = thread_counter_++;
		int num_worker = num_clients_ * num_worker_threads_;
		petuum::PSTableGroup::RegisterThread();
		LOG(INFO) << "client " << client_id_ << ", thread " 
			<< thread_id << " registers!";
			
		// Get w table
		petuum::Table<float> w_table = petuum::PSTableGroup::GetTableOrDie<float>(0);
		//register row
		if(thread_id == 0)
			w_table.GetAsyncForced(0);
		petuum::PSTableGroup::GlobalBarrier();
		
		float temp,error;
		std::string A_data_file,b_data_file,sol_data_file,output_file;
		char Adir[100],bdir[100],outputdir[100];
		std::vector<float> w_cache;
		boost::posix_time::ptime start,end;
		boost::posix_time::time_duration difftime;
		int networktime,sumTime = 0;
		
		sprintf(Adir,"A%d.dat",client_id_*num_worker_threads_+thread_id);
		sprintf(bdir,"b%d.dat",client_id_*num_worker_threads_+thread_id);
		sprintf(outputdir,"objvalue_%d.txt",table_staleness_);
		
		A_data_file = data_dir + Adir;
		b_data_file = data_dir + bdir;
		sol_data_file = data_dir + "solution.dat";
		output_file = output_dir + outputdir;
		
		FILE *fpA = fopen(A_data_file.c_str(),"r");
		FILE *fpb = fopen(b_data_file.c_str(),"r");
		FILE *fps = fopen(sol_data_file.c_str(),"r");
		std::ofstream fout(output_file);
		
		if(!fpA && !fpb && !fps){
			LOG(INFO) << "open the source file error!!!";
		}
		
		//source data
		Eigen::MatrixXf A(row,feature);
		Eigen::VectorXf b(row);
		Eigen::VectorXf s(feature);
		Eigen::MatrixXf lemon(feature,feature);
		Eigen::MatrixXf lemonI(feature,feature);
		Eigen::MatrixXf identity(feature,feature);
		Eigen::VectorXf result(feature);
		Eigen::VectorXf obj(row);
		
		//parameters
		Eigen::VectorXf x(feature);
		Eigen::VectorXf x_diff(feature);
		Eigen::VectorXf z(feature);
		Eigen::VectorXf y(feature);
		Eigen::VectorXf w(feature);
		Eigen::VectorXf w_old(feature);
		Eigen::VectorXf w_diff(feature);
		
		x.setZero();
		z.setZero();
		y.setZero();
		w.setZero();
		w_old.setZero();
		w_diff.setZero();
		
		//init the z_table
		if(thread_id == 0 && client_id_ == 0){
			petuum::UpdateBatch<float> w_update;
			for(int i = 0;i < feature;i++){
				w_update.Update(i,w(i));
			}
			w_table.BatchInc(0, w_update);
		}
		//load data from file
		for(int i = 0;i < row;i++){
			for(int j=0;j < feature;j++){
				fscanf(fpA,"%f",&temp);
				A(i,j) = temp;
			}
		}
		for(int i = 0;i < row;i++){
			fscanf(fpb,"%f",&temp);
			b(i) = temp;
		}
		
		for(int i = 0;i < feature;i++){
			fscanf(fps,"%f",&temp);
			s(i) = temp;
		}
		
		fclose(fpA);
		fclose(fpb);
		fclose(fps);
		
		//warm start
		//lemon = A.transpose()  * A + rho * identity.setIdentity();
		//lemonI = lemon.inverse();
		petuum::PSTableGroup::GlobalBarrier();
		
		//begin to iteration
		for(int iter = 0 ;iter < num_epochs_;iter++){
			//get w from server
			if(iter != 0){
				petuum::RowAccessor row_acc;
				const auto & row = w_table.Get<petuum::DenseRow<float> >(0, &row_acc);
				row.CopyToVector(&w_cache);
				for (int col_id = 0; col_id < feature; ++col_id) {
					w(col_id) = w_cache[col_id];
				}
			}
			
			//compute time
			start = boost::posix_time::microsec_clock::local_time();
			
			//soft threshold
			soft_threshold((1.0/(rho*num_worker) * w),z,lambda/(rho*num_worker));
			
			//update x
			lemon = A.transpose()  * A + rho * identity.setIdentity();
			lemonI = lemon.inverse();
			x = lemonI * (A.transpose() * b + rho * z - y);
			
			//update y
			y = y + rho * (x - z);
			
			//update w	
			w = rho * x + y;
			
			//w diff
			w_diff = w - w_old;
			w_old = w;
			
			//primal error
			x_diff = x - s;
			error = x_diff.norm();
			
			//obj value
			obj = A * x - b;
			
			end = boost::posix_time::microsec_clock::local_time();
			difftime = (end - start);
			sumTime += difftime.total_milliseconds();
			
			//update new z_diff to server
			petuum::UpdateBatch<float> w_update;
			for(int i = 0;i < feature;i++){
				w_update.Update(i,w_diff(i));
			}
			w_table.BatchInc(0, w_update);
			//clock
			petuum::PSTableGroup::Clock();
			
			if(thread_id == 0 && client_id_ == 0){
				LOG(INFO) << "iter: " << iter << ", client " 
					<< client_id_ << ", thread " << thread_id <<
						" primal error: " << error << " object value: "<< 1.0/2 * obj.norm();
				fout << 1.0/2 * obj.norm() << std::endl;
			}
			if(error < errorthreshold||iter == num_epochs_ - 1)
			{
				if(thread_id == 0 && client_id_ == 0){
					boost::posix_time::time_duration runTime = 
						boost::posix_time::microsec_clock::local_time() - initT_;
					networktime = runTime.total_milliseconds() - sumTime;
					LOG(INFO) << "========================== result ===========================";
					LOG(INFO) << "Elapsed time is: "<< runTime.total_milliseconds() << " ms.";
					LOG(INFO) << "Network waiting time is: " << networktime <<" ms.";
					LOG(INFO) << "Compute time is: "<< sumTime << " ms.";
					LOG(INFO) << "=============================================================";
				}
				fout.close();
				return;
			}
		}
	}
	
	LinearRegression::~LinearRegression() {
	}
} // namespace LR
