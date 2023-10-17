functions {
  // Compute log likelihood under a Rasch partial-credit model
  real pcm_lpmf(int y, vector theta_raw) {
    int K = num_elements(theta_raw);
    vector[K+1] theta = append_row(0, cumulative_sum(theta_raw));
    return categorical_logit_lpmf(y + 1 | theta);
  }

  // Random responses under a Rasch partial-credit model
  int pcm_rng(vector theta_raw) {
    int K = num_elements(theta_raw);
    vector[K+1] theta = append_row(0, cumulative_sum(theta_raw));
    return categorical_logit_rng(theta) - 1;
  }

  // Partial sum function for parallel processing
  real partial_log_lik(array[] int partial_xx,
                       int start, int end,
                       array[] int nn, array[] int ii,
                       array[] vector alpha,
                       vector beta, array[] vector delta) {
    // There must be a more efficient way to extra the dimensions, but dims()
    // is not working reliably as of Stan 2.31.
    int K = size(delta[1]);
    int M = size(partial_xx);
    int A1 = size(alpha);
    int A2 = size(alpha[1]);
    array[M] int partial_nn = nn[start:end];
    array[M] int partial_ii = ii[start:end];
    vector[M] log_lik;
    for (m in 1:M) {
      int n = partial_nn[m];
      int i = partial_ii[m];
      vector[K] theta;
      // partial credit model
      if (A1 == 1 && A2 == 1) {
        theta = alpha[1, 1] * (beta[n] - delta[i]);
      // generalised partial credit model
      } else if (A2 == 1) {
        theta = alpha[i, 1] * (beta[n] - delta[i]);
      // extended partial credit model
      } else {
        theta = alpha[i] .* (beta[n] - delta[i]);
      }
      log_lik[m] = pcm_lpmf(partial_xx[m] | theta);
    }
    return sum(log_lik);
  }
}

data {
  int<lower=1> N;                     // number of songs
  int<lower=1> I;                     // number of criteria
  int<lower=1> K;                     // max score on rating scale
  int<lower=1> M;                     // number of observations
  array[M] int<lower=1, upper=N> nn;  // song for observation o
  array[M] int<lower=1, upper=I> ii;  // criterion for observation o
  array[M] int<lower=0, upper=K> xx;  // score for observation o
  int<lower=0, upper=2> ndim_alpha;   // model form
                                      //   0: partial credit
                                      //   1: generalised partial credit
                                      //   2: extended partial credit
}

transformed data {
  int<lower=1, upper=I> A1 = {1, I, I}[ndim_alpha + 1];  // dims(alpha)[1]
  int<lower=1, upper=K> A2 = {1, 1, K}[ndim_alpha + 1];  // dims(alpha)[2]
  array[N] int<lower=0> NN = zeros_int_array(N);  // number of obs per song
  array[I] int<lower=0> II = zeros_int_array(I);  // number of obs per criterion
  for (m in 1:M) {
    NN[nn[m]] += 1;
    II[ii[m]] += 1;
  }
}

parameters {
  vector[K] mu;                         // intercepts per threshold
  vector<lower=0>[K] sigma_delta;       // scales of criterion thresholds
  array[A1] vector<lower=0>[A2] alpha;  // discrimination
  vector[N] beta;                       // song difficulty (always standardised)
  array[I] vector[K] z_delta;           // standardised criterion thresholds
}

transformed parameters {
  array[I] vector[K] delta;              // criterion thresholds
  for (i in 1:I) {
    delta[i] = mu + sigma_delta .* z_delta[i];
  }
}

model {
  mu ~ std_normal();
  sigma_delta ~ exponential(1);
  for (i in 1:A1) {
    alpha[i] ~ exponential(1);
  }
  beta ~ std_normal();
  for (i in 1:I) {
    z_delta[i] ~ std_normal();
  }
  target += reduce_sum(partial_log_lik, xx, 1, nn, ii, alpha, beta, delta);
}

generated quantities {
  vector[N] song_difficulties = 5 + 2 * beta;
  vector[N] song_prior;
  vector[N] song_outfit = zeros_vector(N);
  vector[N] song_infit;

  array[I] vector[K] criterion_thresholds;
  array[I] vector[K] criterion_prior;
  vector[I] criterion_outfit = zeros_vector(I);
  vector[I] criterion_infit;

  vector[M] log_lik;
  array[M] int<lower=0> xx_rep;

  for (n in 1:N) {
    song_prior[n] = normal_rng(5, 2);
  }

  for (i in 1:I) {
    criterion_thresholds[i] = 5 + 2 * delta[i];
    criterion_prior[i] = 5 + 2 * to_vector(normal_rng(mu, sigma_delta));
  }

  {
    vector[N] song_log_lik = zeros_vector(N);
    vector[N] song_entropy = zeros_vector(N);
    vector[I] criterion_log_lik = zeros_vector(I);
    vector[I] criterion_entropy = zeros_vector(I);
    for (m in 1:M) {
      int i = ii[m];
      int n = nn[m];
      vector[K] theta;
      vector[K + 1] log_p;
      real entropy;
      real outfit;
      // partial credit model
      if (A1 == 1 && A2 == 1) {
        theta = alpha[1, 1] * (beta[n] - delta[i]);
      // generalised partial credit model
      } else if (A2 == 1) {
        theta = alpha[i, 1] * (beta[n] - delta[i]);
      // extended partial credit model
      } else {
        theta = alpha[i] .* (beta[n] - delta[i]);
      }
      for (k in 1:(K + 1)) {
        log_p[k] = pcm_lpmf(k - 1 | theta);
      }
      log_lik[m] = log_p[xx[m] + 1];
      entropy = -dot_product(exp(log_p), log_p);
      outfit = -log_lik[m] / entropy;
      xx_rep[m] = pcm_rng(theta);
      song_log_lik[n] += log_lik[m];
      song_entropy[n] += entropy;
      song_outfit[n] += outfit;
      criterion_log_lik[i] += log_lik[m];
      criterion_entropy[i] += entropy;
      criterion_outfit[i] += outfit;
    }
    song_outfit = song_outfit ./ to_vector(NN);
    song_infit = -song_log_lik ./ song_entropy;
    criterion_outfit = criterion_outfit ./ to_vector(II);
    criterion_infit = -criterion_log_lik ./ criterion_entropy;
  }
}
