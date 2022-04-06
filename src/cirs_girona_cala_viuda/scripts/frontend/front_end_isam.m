clear; close all; clc;
format long;
import gtsam.*;
fileIEKF = 'E:\US\Umich_Study\ROB530_MR\Project\UW_SLAM-master\UW_SLAM-master\src\cirs_girona_cala_viuda\data\states.csv';
fileIMU = 'E:\US\Umich_Study\ROB530_MR\Project\UW_SLAM-master\UW_SLAM-master\src\cirs_girona_cala_viuda\data\full_dataset\imu_adis_ros.csv';
fileStTime = 'E:\US\Umich_Study\ROB530_MR\Project\UW_SLAM-master\UW_SLAM-master\src\cirs_girona_cala_viuda\data\state_times.csv';
noiseModelPosr = noiseModel.Diagonal.Precisions(0.1*[1;1;1;1;1;1]);
firstPose = 2;
StatesSkip = 1; % Skip this many States measurements each time

%% Get initial conditions for the estimated trajectory
[states,state_times] = read_iekf_states(fileIEKF,fileStTime);
first_p = gtsam.Point3(states.x(firstPose), states.y(firstPose), states.z(firstPose));
first_rot = gtsam.Rot3.Ypr(states.psi(firstPose),states.theta(firstPose), states.phi(firstPose));
states.Position = [];
for i = 1:numel(state_times)
    states.Position{i} = gtsam.Point3(states.x(i), states.y(i), states.z(i));
end
%currentVelocityGlobal = LieVector([states.u(firstPose);states.v(firstPose);states.r(firstPose)]);
currentVelocityGlobal = LieVector([0;0;0]);
% currentVelocityGlobal = gtsam.Point3(states.u(firstPose), states.v(firstPose), states.r(firstPose)); % the vehicle is stationary at the beginning
currentPoseGlobal = gtsam.Pose3(first_rot, first_p); % initial pose is the reference frame (navigation frame)
currentBias = imuBias.ConstantBias(zeros(3,1), zeros(3,1));
sigma_init_x = noiseModel.Isotropic.Precisions([ 0.0; 0.0; 0.0; 1; 1; 1 ]);
sigma_init_v = noiseModel.Isotropic.Sigma(3, 1000.0);
sigma_init_b = noiseModel.Isotropic.Sigmas([ 0.100; 0.100; 0.100; 5.00e-05; 5.00e-05; 5.00e-05 ]);
acc_bias = [0.067; 0.115; 0.320];
gyro_bias = [0.067; 0.115; 0.320];
AccelerometerBiasSigma =  1.6700e-04;
GyroscopeBiasSigma = 2.9100e-06;
sigma_between_b = [AccelerometerBiasSigma * ones(3,1); GyroscopeBiasSigma * ones(3,1) ]; % BiasSigma
accSigma = 0.0100;
gyroSigma = 1.7500e-04;
integrationSigma = 0;
g = [0;0;-9.8];
w_coriolis = [0;0;0];
currentBias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias);
%% Solver object
isamParams = ISAM2Params;
isamParams.setFactorization('CHOLESKY');
isamParams.setRelinearizeSkip(10);
isam = gtsam.ISAM2(isamParams);
newFactors = NonlinearFactorGraph;
newValues = Values;

%% Main loop:
% (1) we read the measurements
% (2) we create the corresponding factors in the graph
% (3) we solve the graph to obtain and optimal estimate of robot trajectory
IMU = read_imu(fileIMU);

disp('-- Starting main loop: inference is performed at each time step, but we plot trajectory every 10 steps')

for measurementIndex = firstPose:size(states.x,2)
    
    % At each non=IMU measurement we initialize a new node in the graph
    currentPoseKey = symbol('x',measurementIndex);
    currentVelKey =  symbol('v',measurementIndex);
    currentBiasKey = symbol('b',measurementIndex);
    t = state_times(measurementIndex);
    
    if measurementIndex == firstPose
        %% Create initial estimate and prior on initial pose, velocity, and biases
        newValues.insert(currentPoseKey, currentPoseGlobal);
        newValues.insert(currentVelKey, currentVelocityGlobal);
        newValues.insert(currentBiasKey, currentBias);
        newFactors.add(PriorFactorPose3(currentPoseKey, currentPoseGlobal, sigma_init_x));
        newFactors.add(PriorFactorLieVector(currentVelKey, currentVelocityGlobal, sigma_init_v));
        newFactors.add(PriorFactorConstantBias(currentBiasKey, currentBias, sigma_init_b));
    else
        t_previous = state_times(measurementIndex-1);
        %% Summarize IMU data between the previous IEKF measurement and now
        IMUindices = find(IMU.times >= t_previous & IMU.times <= t);
        
        currentSummarizedMeasurement = gtsam.ImuFactorPreintegratedMeasurements( ...
            currentBias, accSigma.^2 * eye(3), ...
            gyroSigma.^2 * eye(3), integrationSigma.^2 * eye(3));
        
        if ~isempty(IMUindices)
            for imuIndex = IMUindices
                accMeas = [ IMU.ax(imuIndex); IMU.ay(imuIndex); IMU.az(imuIndex) ];
                omegaMeas = [ IMU.omegaX(imuIndex); IMU.omegaY(imuIndex); IMU.omegaZ(imuIndex)];
                deltaT = IMU.dt(imuIndex);
                currentSummarizedMeasurement.integrateMeasurement(accMeas, omegaMeas, deltaT);
            end
        else
            d=1
%             accMeas = [ IMU.ax(measurementIndex); IMU.ay(measurementIndex); IMU.az(measurementIndex) ];
%             omegaMeas = [ IMU.omegaX(measurementIndex); IMU.omegaY(measurementIndex); IMU.omegaZ(measurementIndex)];
%             deltaT = IMU.dt(measurementIndex);
%             currentSummarizedMeasurement.integrateMeasurement(accMeas, omegaMeas, deltaT);
        end


    % Create IMU factor
    newFactors.add(ImuFactor( ...
      currentPoseKey-1, currentVelKey-1, ...
      currentPoseKey, currentVelKey, ...
      currentBiasKey, currentSummarizedMeasurement, g, w_coriolis));
        
        % Bias evolution as given in the IMU data
        newFactors.add(BetweenFactorConstantBias(currentBiasKey-1, currentBiasKey, imuBias.ConstantBias(zeros(3,1), zeros(3,1)), ...
            noiseModel.Diagonal.Sigmas(sqrt(numel(IMUindices)) * sigma_between_b)));
        
        % Create state factor
        statesPose = Pose3(currentPoseGlobal.rotation, states.Position{measurementIndex});
        if mod(measurementIndex, StatesSkip) == 0
            newFactors.add(PriorFactorPose3(currentPoseKey, statesPose, noiseModelPosr));
        end
        
        % Add initial value
        newValues.insert(currentPoseKey, statesPose);
        newValues.insert(currentVelKey, currentVelocityGlobal);
        newValues.insert(currentBiasKey, currentBias);
        
        % Update solver
        % =======================================================================
        % We accumulate 2*states skip states before updating the solver at
        % first so that the heading becomes observable.
        if measurementIndex > firstPose + 2*StatesSkip
            isam.update(newFactors, newValues);
            newFactors = NonlinearFactorGraph;
            newValues = Values;
            
            if rem(measurementIndex,10)==0 % plot every 10 time steps
                cla;
                plot3DTrajectory(isam.calculateEstimate, 'g-');
                title('Estimated trajectory using ISAM2 (IMU)')
                xlabel('[m]')
                ylabel('[m]')
                zlabel('[m]')
                axis equal
                drawnow;
            end
            % =======================================================================
            currentPoseGlobal = isam.calculateEstimate(currentPoseKey);
            currentVelocityGlobal = isam.calculateEstimate(currentVelKey);
            currentBias = isam.calculateEstimate(currentBiasKey);
        end
    end
    
end % end main loop

%% Read IMU data
function [IMU] = read_imu(fileIMU)
disp('-- Reading IMU data')
IMU_Data = readcell(fileIMU);      % Read Data
IMU.times = []; IMU.dt = IMU_Data{2,1}; IMU.qx = []; IMU.qy = []; IMU.qz = []; IMU.qw = [];
IMU.ax = []; IMU.ay = []; IMU.az = []; IMU.omegaX = []; IMU.omegaY = []; IMU.omegaZ = [];
for row = 2:size(IMU_Data,1)
    %Time
    timeCol = 1;
    IMU.times = [IMU.times IMU_Data{row,timeCol}];
    if (row+1) <= size(IMU_Data,1)
        IMU.dt = [IMU.dt (IMU_Data{row+1,timeCol} - IMU_Data{row,timeCol})];
    end
    % Orientation
    qxCol = 4; qyCol = 5; qzCol = 6; qwCol = 7;
    IMU.qx = [IMU.qx IMU_Data{row,qxCol}]; IMU.qy = [IMU.qy IMU_Data{row,qyCol}]; IMU.qz = [IMU.qz IMU_Data{row,qzCol}]; IMU.qw = [IMU.qw IMU_Data{row,qwCol}];
    % Linear Acceleration
    axCol = 29; ayCol = 30; azCol = 31;
    IMU.ax = [IMU.ax IMU_Data{row,axCol}]; IMU.ay = [IMU.ay IMU_Data{row,ayCol}]; IMU.az = [IMU.az IMU_Data{row,azCol}];
    % Angular Velocity
    omegaxCol = 17; omegayCol = 18; omegazCol = 19;
    IMU.omegaX = [IMU.omegaX IMU_Data{row,omegaxCol}]; IMU.omegaY = [IMU.omegaY IMU_Data{row,omegayCol}]; IMU.omegaZ = [IMU.omegaZ IMU_Data{row,omegazCol}];
    % Acceleration Covariance
    IMU.accCov{row-1} = [IMU_Data{row,32}, IMU_Data{row,33}, IMU_Data{row,34}; IMU_Data{row,35}, IMU_Data{row,36}, IMU_Data{row,37}; IMU_Data{row,38}, IMU_Data{row,39}, IMU_Data{row,40}];
    % Omega Covariance
    IMU.omegaCov{row-1} = [IMU_Data{row,20}, IMU_Data{row,21}, IMU_Data{row,22}; IMU_Data{row,23}, IMU_Data{row,24}, IMU_Data{row,25}; IMU_Data{row,26}, IMU_Data{row,27}, IMU_Data{row,28}];
end
IMU.times = IMU.times.*1e-9;
IMU.dt = IMU.dt.*1e-9;
end

%% Read IEKF States

function [states,state_times] = read_iekf_states(fileIEKF,fileStTime)
disp('-- Reading IEKF data')
IEKF_Data = readcell(fileIEKF);      % Read Data
IEKF_time = readcell(fileStTime);      % Read Data
states.x = []; states.y = []; states.z = []; states.u = []; states.v = []; states.r = [];
states.phi = []; states.theta = []; states.psi = [];
state_times = [];
for row = 2:size(IEKF_Data,1)
    states.x = [states.x IEKF_Data{row,7}]; states.y = [states.y IEKF_Data{row,8}]; states.z = [states.z IEKF_Data{row,9}];   %p_x, p_y, p_z
    states.u = [states.u IEKF_Data{row,4}]; states.v = [states.v IEKF_Data{row,5}]; states.r = [states.r IEKF_Data{row,6}];   %v_x, v_y, v_z
    states.phi = [states.phi IEKF_Data{row,1}]; states.theta = [states.theta IEKF_Data{row,2}]; states.psi = [states.psi IEKF_Data{row,3}];   %theta_x, theta_y, theta_z
    state_times = [state_times  IEKF_time{row,1}];
end
state_times = state_times.*1e-9;
end
