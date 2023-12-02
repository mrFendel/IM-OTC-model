---
tags:
  - mustdo
  - build
  - quant
  - num
  - moex
---
[CppMonk Tutorial](https://www.youtube.com/playlist?list=PLvKAPIGzFEr8n7WRx8RptZmC1rXeTzYtA)
[HABR1](https://habr.com/ru/articles/166693/)
[HABR](https://habr.com/ru/articles/140274/)
[WIKI](https://en.wikipedia.org/wiki/Kalman_filter)

# Smoothing Kalman

**Goal:** 
estimate $x_t \ and \ v_t$
given measurements $Z = x_t + \epsilon_t$
$$
\begin{align}
X_t = \begin{bmatrix}
   x_{t} \\
   v_{t} \\
 \end{bmatrix} \quad
 Z_t = \begin{bmatrix}
   z_{t} \\
 \end{bmatrix}
\end{align}

$$
**Time evolution**
$$ x_{t+1} = x_t + v_t \delta t + \frac{1}{2}a \delta t^2; \quad  v_{t+1} = v_t + a \delta t$$
$$ X_{t+1} = \begin{bmatrix}
  1 & \delta t \\
  0 & 1 \\
 \end{bmatrix} X_t 
 \ + \ 
 \begin{bmatrix}
  \frac{1}{2}\delta t^2 \\
  \delta t \\
 \end{bmatrix} = F \cdot X_t + G \cdot a $$
 **Assumptions:** acceleration and error are noise.
	$$ a, \epsilon \ - \ N(0, \Sigma)$$
$$ 
Z_t = \begin{bmatrix} 1 & 0 \end{bmatrix} X_t + \epsilon_t = H \cdot X_t  + \epsilon_t
$$

**Prediction step**
$$
X_{t} - N(X_t, P_t)
$$
$$
X_{t+1} = F X_t; \quad
P_{t+1} = F P_t F^T + G \Sigma^2_a G^T
$$
**Measurement Step**
$$Y = Z_t - H \cdot X_t \ \text{- error between measurement and prediction}$$
$$S_t = H \cdot P_t \cdot H^T + \Sigma^2_a \text{ - error of covariance estimate}$$
$$ K = P_t \cdot H^T \cdot S^{-1}_t \text{ - optimal Kalman step}$$
$$ X^{udated}_t = X_t + K \cdot Y \text{ - updated location step} $$
$$ P^{updated}_t = [I - K \cdot H] \cdot P_t \text{ - updated covariance} $$


# Cointegration Kalman

**Goal:** 
estimate $x_t \ and \ v_t$ - intercept and $\frac{\sigma_2}{\sigma_1} corr$
assuming that $r_2 = x + v \cdot r_1 + \epsilon^{\prime}$
$$
\begin{align}
X_t = \begin{bmatrix}
   x_{t} \\
   v_{t} \\
 \end{bmatrix} \quad
 Z_t = \begin{bmatrix}
   z_{t} \\
 \end{bmatrix} = r^2_t
\end{align}

$$

**Transition model**
$$ X_{t+1} = F \cdot X_t + \epsilon_x$$
$$ F = \begin{bmatrix}
  1 & 0 \\
  0 & 1 \\
 \end{bmatrix}; 
 \quad \epsilon_x - N(0, Q); 
 \quad Q = \begin{bmatrix}
  \delta_{x} & 0 \\
  0 & \delta_{v} \\
 \end{bmatrix} $$
 
 **Observation Model**	
$$ 
r^2_t = Z_t = H \cdot X_t  + \epsilon_t
$$
$$ H = [1 \quad r^1_t]; \quad \epsilon_x - N(0, R);  \quad R = \begin{bmatrix}
  1 & 0 \\
  0 & 1 \\
 \end{bmatrix}$$
 
**Prediction step**
$$
X_{t} - N(X_t, P_t)
$$
$$
X_{t+1} = F X_t; \quad
P_{t+1} = F P_t F^T + Q
$$

**Measurement Step**
$$Y = Z_t - H \cdot X_t \ \text{- error between measurement and prediction}$$
$$S_t = H \cdot P_t \cdot H^T + R \text{ - error of covariance estimate}$$
$$ K = P_t \cdot H^T \cdot S^{-1}_t \text{ - optimal Kalman step}$$
$$ X^{udated}_t = X_t + K \cdot Y \text{ - updated location step} $$
$$ P^{updated}_t = [I - K \cdot H] \cdot P_t \text{ - updated covariance} $$

