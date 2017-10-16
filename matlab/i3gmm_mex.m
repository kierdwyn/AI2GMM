%CLASS_INTERFACE Example MATLAB class wrapper to an underlying C++ class
classdef i3gmm_mex < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
    end
    properties
        % Settings
        sweep;
        burnin;
        prior;
        init_hyperparams;
        
        % Results
        gibbs_score;
        samples;
        sample_last;
        hyperparams;
        
        % data
        xtrain;
        ytrain;
        xtest;
    end
    properties (Constant)
        %-----model control parameters-----
        default_sweep = 400;
        default_burnin = 5;
        default_sample = 1;
        %-----hyper params------
        % sigma0 = diag(diag(cov(x)));
        % mu0 = [36.4603 -119.3265];
        % sigma0 = [21.4972 0;0 23.1351];
        default_conf = [0.1 0.5 22 1 1]; % [kappa0 kappa1 m alpha gamma]
    end
    methods(Static)
        %% Run i3gmm using default configurations
        function [sample, gibbs_score] = run_i3gmm(x, prior, conf,...
                xtrain, ytrain, sweep, init_sweep, burnin, n_sample,...
                logfname)
            if ~exist('sweep', 'var'), sweep = i3gmm.sweep; end
            if ~exist('init_sweep', 'var'), init_sweep = 0; end
            if exist('prior', 'var')
                i3gmm_obj = i3gmm(prior', conf);
            else
                i3gmm_obj = i3gmm(x');
            end
            if exist('xtrain','var')
                add_prior(i3gmm_obj, xtrain', ytrain);
                adjust_weights(i3gmm_obj, ones(size(ytrain))*(1/numel(ytrain)));
                cluster_gibbs(i3gmm_obj, init_sweep);
                prior_like(i3gmm_obj);
            end
            add_data(i3gmm_obj, x');
            [gibbs_score, sample] = cluster_gibbs(i3gmm_obj, sweep,...
                burnin, n_sample, logfname);
            fprintf('finished.\n');
        end
    end
    methods
        %% Constructor - Create a new i3gmm instance 
        function this = i3gmm_mex(varargin)
            if nargin == 1
                % varargin{1}: x, the data set. Create an i3gmm obj using
                % default configurations and add the data set.
                x = varargin{1};
                mu0 = mean(x);
                sigma0 = cov(x);
                this.init_hyperparams = i3gmm.default_conf;
                this.prior = [mu0;sigma0]';
                this.objectHandle = i2gmm_semi('new', this.prior, i3gmm.conf);
                add_data(this, x);
            else
                this.objectHandle = i2gmm_semi('new', varargin{:});
                this.prior = varargin{1};
                this.init_hyperparams = varargin{2};
            end
        end
        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            i2gmm_semi('delete', this.objectHandle);
        end

        %% Add_data - an example class method call
        function varargout = add_data(this, varargin)
            [varargout{1:nargout}] = i2gmm_semi('add_data', this.objectHandle, varargin{:});
            this.xtest = varargin{1};
        end
        %% Add training data and labels.
        function varargout = add_prior(this, varargin)
            [varargout{1:nargout}] = i2gmm_semi('add_prior', this.objectHandle, varargin{:});
            this.xtrain = varargin{1};
            this.ytrain = varargin{2};
        end
        %% Cluster_gibbs - another example class method call
        function varargout = cluster_gibbs(this, varargin)
            if nargin == 1
                % Using default configuration
                [varargout{1:nargout}] = i2gmm_semi('cluster_gibbs',this.objectHandle,...
                    i3gmm.default_sweep, 'result_gibbslog.txt', i3gmm.default_burnin,...
                    i3gmm.default_sample);
                this.sweep = i3gmm.default_sweep;
                this.burnin = i3gmm.default_burin;
            else
                [varargout{1:nargout}] = i2gmm_semi('cluster_gibbs', this.objectHandle, varargin{:});
                this.sweep = varargin{1};
                if nargin > 2, this.burnin = varargin{2}; end
            end
            this.gibbs_score = [this.gibbs_score varargout{1}];
            if nargout > 1, this.samples = varargout{2}; end;
        end
        %% adjust_weights - another example class method call
        function varargout = adjust_weights(this, varargin)
            [varargout{1:nargout}] = i2gmm_semi('adjust_weights', this.objectHandle, varargin{:});
        end
        %% prior_like - Calculate likelihood for each training point
        function prior_like(this)
            i2gmm_semi('prior_like', this.objectHandle);
        end
        %% get_current_labels - another example class method call
        function label = get_current_labels(this, varargin)
            label = i2gmm_semi('get_current_labels', this.objectHandle, varargin{:});
            label = relabel(label(:,1));
            this.sample_last = label;
        end
        %% Get hyper parameters
        function hyper = get_hyperparams(this)
            hyper = i2gmm_semi('get_hyperparams', this.objectHandle);
            this.hyperparams = hyper;
        end
        %% Get likelihood for all customers
        function llike = get_likelihood(this)
            llike = i2gmm_semi('get_likelihood', this.objectHandle);
        end
    end
end