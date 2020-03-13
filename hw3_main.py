import numpy as np
from utils import *
from scipy import linalg

if __name__ == '__main__':
        filename = "./data/0027.npz"
        t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
        features = features[:,0:np.size(features,1):10,:]       # landmarks downsampled to every 10th feature

        rotational_velocity = -rotational_velocity      ### CHANGE: if data 0027, uncomment for correct visualization
        
        # K: left camera intrinsic matrix
        # [fx 0 cx
        #  0 fy cy
        #  0  0  1]
        # b: stereo camera baseline
        # cam_T_imu: imu to left camera extrinsic matrix in SE(3)
        # [ 0 -1  0 t1
        #   0  0 -1 t2
        #   1  0  0 t3
        #   0  0  0  1]

        # ----------------------------------- INITIALIZE VARIABLES ---------------------------------- #
        samples = len(t[0])
        mu_imu = np.identity(4)                                 # mean 4x4 identity
        sigma_imu = np.identity(6)                              # covariance 6x6 identity
        car_path = np.zeros((4,4,samples))                      # imu trajectory over time
        car_path[:,:,0] = mu_imu                                # save initial position
        M = np.zeros((4,4))                                     # stereo camera calibration matrix (L13,S2)
        M[0:2,0:3] = K[0:2,:]
        M[2:4,0:3] = K[0:2,:]
        M[2,3] = -K[0,0]*b

        V = 10^-4                                               # scalar for covariance noise
        mu_landmark = -1*np.ones((4,np.size(features,1)))       # mean of observation 4xM of -1
        sigma_landmark = np.identity(3*np.size(features,1))*V   # covariance of observation noise 3Mx3M identity (L13,S8)
        D = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
        D = np.kron(np.eye(np.size(features,1)), D)

        # ------------------------------------------------------------------------------------------- #

        for i in range(samples):

                samples += 1    # start at 1
                
                # (a) IMU Localization via EKF Prediction (L13,S13,16)
                
                v_t_col = linear_velocity[:,i].reshape(3,1)             # reshape v_t 1x3 to 3x1
                w_t_col = rotational_velocity[:,i].reshape(3,1)         # reshape w_t 1x3 to 3x1
                u_t = np.vstack((v_t_col, w_t_col))                     # u_t = [v_t; w_t]

                w_t_hat = np.array([[0, -w_t_col[2], w_t_col[1]],       # skew symmetric version of w_t_col
                           [w_t_col[2], 0,          -w_t_col[0]],
                           [-w_t_col[1], w_t_col[0],          0]])
                zero_T = np.zeros((1,3))                                # 0^T, 1x3 zeros
                u_t_hat = np.block([[w_t_hat, v_t_col],[zero_T, 0]])    # u_t_hat = [w_t_hat, v_t; 0^T, 0]      

                v_t_hat = np.array([[0, -v_t_col[2], v_t_col[1]],       # skew symmetric version of v_t_col
                           [v_t_col[2], 0,          -v_t_col[0]],
                           [-v_t_col[1], v_t_col[0],         0]])
                zeros = np.zeros((3,3))                                 # 3x3 zeros
                u_t_adjoint = np.block([[w_t_hat, v_t_hat],[zeros, w_t_hat]]) # u_t with curly hat above

                tau = t[0,i] - t[0,i-1]                                 # time discretization
                
                # Mean Prediction: mu(t+1|t)
                mu_imu = np.dot(linalg.expm(-tau*u_t_hat), mu_imu)
                # Covariance Prediction: sigma(t+1|t)
                W = tau*tau*np.diag(np.random.normal(0,1,6))            # motion model noise
                sigma_imu = np.dot(np.dot(linalg.expm(-tau*u_t_adjoint),sigma_imu), np.transpose(linalg.expm(-tau*u_t_adjoint))) + W

                # Save Robot Path
                car_path[:,:,i] = linalg.inv(mu_imu)

                # (b) Landmark Mapping via EKF Update (L13,S18,19)

                features_i = features[:,:,i]                            # features at this sample
                                                                        # features dim: 17474800 x 3950 x 1106
                feature_i_sum = np.sum(features_i[:,:],0)                       # sum of feature rows
                features_obs_index = np.array(np.where(feature_i_sum!=-4))      # if not [-1 -1 -1 -1]^T, feature is observable, save it

                features_obs_index_new = np.zeros(0)
                features_obs_index_new.dtype=int
                
                features_obs_count = np.size(features_obs_index)        # number of observable features at this sample

                features_obs_new = np.empty((4,0))                      

                T_worldtocam = np.dot(cam_T_imu, mu_imu)                # world to camera transform
                T_camtoworld = linalg.inv(T_worldtocam)

                if (np.size(features_obs_index)!=0):

                        # observable features at this sample
                        features_obs_coords = features_i[:,features_obs_index].reshape(4,features_obs_count)

                        # pixel to world coordinates
                        features_obs = np.ones((4,np.shape(features_obs_coords)[1]))
                        features_obs[0,:] = (features_obs_coords[0,:]-M[0,2])*b/(features_obs_coords[0,:]-features_obs_coords[2,:])
                        features_obs[1,:] = (features_obs_coords[1,:]-M[1,2])*(-M[2,3])/(M[1,1]*(features_obs_coords[0,:]-features_obs_coords[2,:]))
                        features_obs[2,:] = -M[2,3]/(features_obs_coords[0,:]-features_obs_coords[2,:])
                        features_obs = np.dot(T_camtoworld, features_obs)
                        
                        for j in range(features_obs_count):

                                k = features_obs_index[0,j]             # observable feature index

                                # if first time observable feature seen, add to landmark means
                                if (np.array_equal(mu_landmark[:,k],[-1,-1,-1,-1])):
                                        mu_landmark[:,k] = features_obs[:,j]

                                # else append observable feature to new features array
                                else: 
                                        features_obs_index_new = np.append(features_obs_index_new,k)
                                        features_obs_new = np.hstack((features_obs_new, features_obs[:,j].reshape(4,1)))
                           
                        if (np.size(features_obs_index_new)!=0):        # if there are new observable features

                                # Prior mean mu(t+1|t)
                                mu_obs_new = mu_landmark[:,features_obs_index_new]
                                mu_obs_new.reshape((4, np.size(features_obs_index_new)))

                                # Find Jacobian H evaluated at mu(t+1|t)
                                total_number_of_features = np.size(features,1)

                                D2 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
                                H = np.zeros((4*np.size(features_obs_index_new),3*total_number_of_features))
                                 
                                for s in range(np.size(features_obs_index_new)):
                                
                                        k = features_obs_index_new[s]                           # projection function and derivative (L13,S5)
                                        
                                        q = np.dot(T_worldtocam, mu_obs_new[:,s])
                                        proj_der = np.array([[1, 0, -q[0]/q[2], 0],
                                                             [0, 1, -q[1]/q[2], 0],
                                                             [0, 0,     0,      0],
                                                             [0, 0, -q[3]/q[2], 1]])
                                        proj_der = proj_der / q[2]
                                        H[s*4:(s+1)*4,k*3:(k+1)*3] = np.dot(np.dot(np.dot(M,proj_der),T_worldtocam),D2) # circle with dot (L13,S18)

                                # Perform the EKF Update

                                # Kalman Gain
                                K = np.dot(np.dot(sigma_landmark,np.transpose(H)),linalg.inv(np.dot(np.dot(H,sigma_landmark),np.transpose(H))+np.identity(4*np.size(features_obs_index_new))*V))
                                # Observation
                                z = features_i[:,features_obs_index_new].reshape((4,np.size(features_obs_index_new)))

                                q = np.dot(T_worldtocam, mu_obs_new)
                                projection = q/q[2,:]
                                z_hat = np.dot(M, projection)

                                # Mean Update: mu_(t+1|t+1)
                                mu_landmark = (mu_landmark.reshape(-1,1,order='F') + np.dot(np.dot(D,K),(z-z_hat).reshape(-1,1,order='F'))).reshape(4,-1,order='F')
                                # Covariance Update: sigma(t+1|t+1)
                                sigma_landmark = np.dot((np.identity(3*np.shape(features)[1])-np.dot(K,H)),sigma_landmark)

        # You can use the function below to visualize the robot pose over time
        visualize_trajectory_2d(car_path,mu_landmark,path_name="Car Path",show_ori=True)

        # (c) Visual-Inertial SLAM (L13,S8,19)    

        # Implement IMU pose update based on stereo camera observation model
        # 1. Doing localization and mapping jointly using one combined filter state U_(t,m)
        #    consists of IMU inverse pose (car_path) and landmark positions (mu_landmark)
        # 2. Joint covariance (sigma_joint) has dimension (3M+6)*(3M+6) and associated mean (mu_joint)
        # 3. Update in one shot while computing 3M+6 perturbation = K*innovation
        # 4. Adding perturbation to previous mean (mu_joint_t-1) can be done separately
        #
        # Piazza posts: @459 @469_f2

##        mu_landmark = -1*np.ones((3*np.size(features,1),1))          # mean of imu 3Mx1
##        mu_imu = np.identity(4)                                      # mean of observation 4x4
##        
##        sigma_joint = np.identity(3*np.size(features,1)+6,3*np.size(features,1)+6))*V   # covariance of observation noise (3M+6)x(3M+6)
##
##        samples += 1    # start at 1
##                
##                # (a) IMU Localization via EKF Prediction (L13,S13,16)
##                
##                v_t_col = linear_velocity[:,i].reshape(3,1)             # reshape v_t 1x3 to 3x1
##                w_t_col = rotational_velocity[:,i].reshape(3,1)         # reshape w_t 1x3 to 3x1
##                u_t = np.vstack((v_t_col, w_t_col))                     # u_t = [v_t; w_t]
##
##                w_t_hat = np.array([[0, -w_t_col[2], w_t_col[1]],       # skew symmetric version of w_t_col
##                           [w_t_col[2], 0,          -w_t_col[0]],
##                           [-w_t_col[1], w_t_col[0],          0]])
##                zero_T = np.zeros((1,3))                                # 0^T, 1x3 zeros
##                u_t_hat = np.block([[w_t_hat, v_t_col],[zero_T, 0]])    # u_t_hat = [w_t_hat, v_t; 0^T, 0]      
##
##                v_t_hat = np.array([[0, -v_t_col[2], v_t_col[1]],       # skew symmetric version of v_t_col
##                           [v_t_col[2], 0,          -v_t_col[0]],
##                           [-v_t_col[1], v_t_col[0],         0]])
##                zeros = np.zeros((3,3))                                 # 3x3 zeros
##                u_t_adjoint = np.block([[w_t_hat, v_t_hat],[zeros, w_t_hat]]) # u_t with curly hat above
##
##                tau = t[0,i] - t[0,i-1]                                 # time discretization
##                
##                # Mean Prediction: mu(t+1|t)
##                mu_imu = np.dot(linalg.expm(-tau*u_t_hat), mu_imu)
##                # Covariance Prediction: sigma(t+1|t)
##                W = tau*tau*np.diag(np.random.normal(0,1,6))            # motion model noise
##                sigma_imu = np.dot(np.dot(linalg.expm(-tau*u_t_adjoint),sigma_imu), np.transpose(linalg.expm(-tau*u_t_adjoint))) + W
##
##                # Save Robot Path
##                car_path[:,:,i] = linalg.inv(mu_imu)
##
##                # (b) Landmark Mapping via EKF Update (L13,S18,19)
##
##                features_i = features[:,:,i]                            # features at this sample
##                                                                        # features dim: 17474800 x 3950 x 1106
##                feature_i_sum = np.sum(features_i[:,:],0)                       # sum of feature rows
##                features_obs_index = np.array(np.where(feature_i_sum!=-4))      # if not [-1 -1 -1 -1]^T, feature is observable, save it
##
##                features_obs_index_new = np.zeros(0)
##                features_obs_index_new.dtype=int
##                
##                features_obs_count = np.size(features_obs_index)        # number of observable features at this sample
##
##                features_obs_new = np.empty((4,0))                      
##
##                T_worldtocam = np.dot(cam_T_imu, mu_imu)                # world to camera transform
##                T_camtoworld = linalg.inv(T_worldtocam)
##
##                if (np.size(features_obs_index)!=0):
##
##                        # observable features at this sample
##                        features_obs_coords = features_i[:,features_obs_index].reshape(4,features_obs_count)
##
##                        # pixel to world coordinates
##                        features_obs = np.ones((4,np.shape(features_obs_coords)[1]))
##                        features_obs[0,:] = (features_obs_coords[0,:]-M[0,2])*b/(features_obs_coords[0,:]-features_obs_coords[2,:])
##                        features_obs[1,:] = (features_obs_coords[1,:]-M[1,2])*(-M[2,3])/(M[1,1]*(features_obs_coords[0,:]-features_obs_coords[2,:]))
##                        features_obs[2,:] = -M[2,3]/(features_obs_coords[0,:]-features_obs_coords[2,:])
##                        features_obs = np.dot(T_camtoworld, features_obs)
##                        
##                        for j in range(features_obs_count):
##
##                                k = features_obs_index[0,j]             # observable feature index
##
##                                # if first time observable feature seen, add to landmark means
##                                if (np.array_equal(mu_landmark[:,k],[-1,-1,-1,-1])):
##                                        mu_landmark[:,k] = features_obs[:,j]
##
##                                # else append observable feature to new features array
##                                else: 
##                                        features_obs_index_new = np.append(features_obs_index_new,k)
##                                        features_obs_new = np.hstack((features_obs_new, features_obs[:,j].reshape(4,1)))
##                           
##                        if (np.size(features_obs_index_new)!=0):        # if there are new observable features
##
##                                # Prior mean mu(t+1|t)
##                                mu_obs_new = mu_landmark[:,features_obs_index_new]
##                                mu_obs_new.reshape((4, np.size(features_obs_index_new)))
##
##                                # Find Jacobian H evaluated at mu(t+1|t) as described in L13,S19
##                                total_number_of_features = np.size(features,1)
##
##                                D2 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
##                                H = np.zeros((4*np.size(features_obs_index_new),3*total_number_of_features))
##                                 
##                                for s in range(np.size(features_obs_index_new)):
##                                
##                                        k = features_obs_index_new[s]                           # projection function and derivative (L13,S5)
##                                        
##                                        q = np.dot(T_worldtocam, mu_obs_new[:,s])
##                                        proj_der = np.array([[1, 0, -q[0]/q[2], 0],
##                                                             [0, 1, -q[1]/q[2], 0],
##                                                             [0, 0,     0,      0],
##                                                             [0, 0, -q[3]/q[2], 1]])
##                                        proj_der = proj_der / q[2]
##                                        H[s*4:(s+1)*4,k*3:(k+1)*3] = np.dot(np.dot(np.dot(M,proj_der),T_worldtocam),D2) # circle with dot (L13,S18)
##
##                                # Find Jacobian H avaluated at mu(t|t) s described in L13,S8
##                                
##                                # Perform the EKF Update
##
##                                # Kalman Gain
##                                K = np.dot(np.dot(sigma_landmark,np.transpose(H)),linalg.inv(np.dot(np.dot(H,sigma_landmark),np.transpose(H))+np.identity(4*np.size(features_obs_index_new))*V))
##                                # Observation
##                                z = features_i[:,features_obs_index_new].reshape((4,np.size(features_obs_index_new)))
##
##                                q = np.dot(T_worldtocam, mu_obs_new)
##                                projection = q/q[2,:]
##                                z_hat = np.dot(M, projection)
##
##                                # Mean Update: mu_(t+1|t+1)
##                                mu_landmark = (mu_landmark.reshape(-1,1,order='F') + np.dot(np.dot(D,K),(z-z_hat).reshape(-1,1,order='F'))).reshape(4,-1,order='F')
##                                # Covariance Update: sigma(t+1|t+1)
##                                sigma_landmark = np.dot((np.identity(3*np.shape(features)[1])-np.dot(K,H)),sigma_landmark)
##
##        # You can use the function below to visualize the robot pose over time
##        visualize_trajectory_2d(car_path,mu_landmark,path_name="Car Path",show_ori=True)

        
        











