%% ========================================================================
%% GIẢI THÍCH CHI TIẾT: ÁP DỤNG GWO VÀO BÀI TOÁN JCAS BEAMFORMING
%% ========================================================================

%% 1. MAPPING BÀI TOÁN -> GWO
% 
% Bài toán JCAS Beamforming:
% ---------------------------
% - Tìm: Vector trọng số W (M×1, số phức)
% - Sao cho: Beam pattern P = W^H · A khớp với pattern mong muốn PdM
% - Fitness: f(W) = Σ|P - PdM| = Σ|W^H·A - PdM|
%
% Ánh xạ sang GWO:
% ----------------
% - Con sói (wolf) = 1 ứng viên vector W
% - Vị trí con sói = Vector W (12 phần tử phức)
% - Alpha wolf = Vector W tốt nhất (fitness nhỏ nhất)
% - Beta wolf = Vector W tốt thứ 2
% - Delta wolf = Vector W tốt thứ 3
% - Omega wolves = Các vector W còn lại

%% 2. KHỞI TẠO QUẦN THỂ SÓI (Complex-valued)
M = 12; % Số phần tử anten
N = 30; % Số sói trong bầy

% Mỗi con sói là 1 vector phức W = a + jb
% Phần thực: a ∈ [-0.5, 0.5]
% Phần ảo: b ∈ [-0.5, 0.5]
Wolves = zeros(N, M);
for i = 1:N
    % Khởi tạo ngẫu nhiên
    Wolves(i, :) = (rand(1, M) - 0.5) + 1j*(rand(1, M) - 0.5);
end

fprintf('=== KHỞI TẠO ===\n');
fprintf('Số sói (Search Agents): %d\n', N);
fprintf('Chiều không gian (Dimensions): %d\n', M);
fprintf('Kiểu dữ liệu: Complex (a + jb)\n');
fprintf('Wolf 1 example: [%.2f%+.2fj, %.2f%+.2fj, ...]\n', ...
    real(Wolves(1,1)), imag(Wolves(1,1)), real(Wolves(1,2)), imag(Wolves(1,2)));

%% 3. ĐÁNH GIÁ FITNESS
% Load parameters giả định
theta = (-90:0.1:90-0.1)*pi/180;
lambda = 1;
Q = 160;
phi = 1;
eqDir = -1:phi/Q:1-phi/Q;

A = generateSteeringVector(theta, M, lambda);
Aq = generateQuantizedArrResponse(M, eqDir);
[PdM, ~, ~] = generateDesPattern(eqDir, sin(0), Aq);
alpha = sort([find(ismember(eqDir, eqDir(1:4:end))), find(PdM)]);

% Tính fitness cho mỗi con sói
fitness = zeros(N, 1);
for i = 1:N
    W = Wolves(i, :)';
    % Pattern thực tế do W tạo ra
    Pattern_actual = abs(W' * Aq(:, alpha));
    % Pattern mong muốn
    Pattern_desired = PdM(:, alpha);
    % Fitness = tổng sai lệch
    fitness(i) = sum(abs(Pattern_actual - Pattern_desired));
end

fprintf('\n=== ĐÁNH GIÁ FITNESS ===\n');
fprintf('Fitness function: f(W) = Σ|W^H·Aq - PdM|\n');
fprintf('Min fitness: %.4f\n', min(fitness));
fprintf('Max fitness: %.4f\n', max(fitness));
fprintf('Mean fitness: %.4f\n', mean(fitness));

%% 4. XẾP HẠNG VÀ CHỌN LÃNH ĐẠO
[sorted_fitness, indices] = sort(fitness);

Alpha_wolf = Wolves(indices(1), :);
Alpha_score = sorted_fitness(1);

Beta_wolf = Wolves(indices(2), :);
Beta_score = sorted_fitness(2);

Delta_wolf = Wolves(indices(3), :);
Delta_score = sorted_fitness(3);

fprintf('\n=== PHÂN CẤP BẦY SÓI ===\n');
fprintf('🐺 Alpha (Best):   Fitness = %.4f\n', Alpha_score);
fprintf('🐺 Beta (2nd):     Fitness = %.4f\n', Beta_score);
fprintf('🐺 Delta (3rd):    Fitness = %.4f\n', Delta_score);
fprintf('🐺 Omega (others): %d wolves\n', N-3);

%% 5. CẬP NHẬT VỊ TRÍ THEO CÔNG THỨC GWO
iter = 1;
Max_iter = 50;
a = 2 - iter * (2 / Max_iter); % a giảm từ 2 -> 0

% Chọn 1 con sói Omega để demo
omega_idx = 10;
W_omega_old = Wolves(omega_idx, :);

fprintf('\n=== CẬP NHẬT VỊ TRÍ (Iteration %d) ===\n', iter);
fprintf('Parameter a = %.4f\n', a);

% Cập nhật theo từng chiều (dimension)
W_omega_new = zeros(1, M);
for d = 1:M
    % --- Bước 1: Tính toán dựa trên Alpha ---
    r1 = rand();
    r2 = rand();
    A1 = 2*a*r1 - a;
    C1 = 2*r2;
    
    % Khoảng cách đến Alpha (có trọng số C1)
    D_alpha = abs(C1 * Alpha_wolf(d) - W_omega_old(d));
    % Vị trí giả định nếu theo Alpha
    X1 = Alpha_wolf(d) - A1 * D_alpha;
    
    % --- Bước 2: Tính toán dựa trên Beta ---
    r1 = rand();
    r2 = rand();
    A2 = 2*a*r1 - a;
    C2 = 2*r2;
    
    D_beta = abs(C2 * Beta_wolf(d) - W_omega_old(d));
    X2 = Beta_wolf(d) - A2 * D_beta;
    
    % --- Bước 3: Tính toán dựa trên Delta ---
    r1 = rand();
    r2 = rand();
    A3 = 2*a*r1 - a;
    C3 = 2*r2;
    
    D_delta = abs(C3 * Delta_wolf(d) - W_omega_old(d));
    X3 = Delta_wolf(d) - A3 * D_delta;
    
    % --- Bước 4: Vị trí mới = Trung bình 3 hướng ---
    W_omega_new(d) = (X1 + X2 + X3) / 3;
    
    if d == 1  % In chi tiết cho dimension đầu tiên
        fprintf('\nDimension %d:\n', d);
        fprintf('  Current position: %.4f%+.4fj\n', real(W_omega_old(d)), imag(W_omega_old(d)));
        fprintf('  Alpha guides to: %.4f%+.4fj (A1=%.2f, C1=%.2f)\n', real(X1), imag(X1), A1, C1);
        fprintf('  Beta guides to:  %.4f%+.4fj (A2=%.2f, C2=%.2f)\n', real(X2), imag(X2), A2, C2);
        fprintf('  Delta guides to: %.4f%+.4fj (A3=%.2f, C3=%.2f)\n', real(X3), imag(X3), A3, C3);
        fprintf('  New position:    %.4f%+.4fj (average)\n', real(W_omega_new(d)), imag(W_omega_new(d)));
    end
end

%% 6. Ý NGHĨA CỦA CÁC THAM SỐ
fprintf('\n=== Ý NGHĨA THAM SỐ ===\n');
fprintf('• Parameter a: %.4f\n', a);
if a > 1
    fprintf('  -> |A| có thể > 1 → EXPLORATION (khám phá rộng)\n');
else
    fprintf('  -> |A| < 1 → EXPLOITATION (khai thác cục bộ)\n');
end

fprintf('\n• Coefficient A = 2*a*r - a:\n');
fprintf('  -> Điều khiển bước nhảy (step size)\n');
fprintf('  -> |A| > 1: Nhảy xa khỏi leader (tìm kiếm mới)\n');
fprintf('  -> |A| < 1: Di chuyển về phía leader (hội tụ)\n');

fprintf('\n• Coefficient C = 2*r:\n');
fprintf('  -> Trọng số ngẫu nhiên cho vị trí con mồi\n');
fprintf('  -> C > 1: Nhấn mạnh vị trí leader\n');
fprintf('  -> C < 1: Giảm ảnh hưởng leader\n');

%% 7. KẾT QUẢ SO SÁNH VỚI ILS
fprintf('\n=== SO SÁNH GWO vs TWO-STEP ILS ===\n');
fprintf('\n Two-Step ILS:\n');
fprintf('  • Deterministic (không ngẫu nhiên)\n');
fprintf('  • Hội tụ nhanh (10-20 iterations)\n');
fprintf('  • Dựa trên Least Squares (giải tích)\n');
fprintf('  • Có thể bị kẹt local optimum\n');

fprintf('\n Standard GWO:\n');
fprintf('  • Stochastic (có yếu tố ngẫu nhiên)\n');
fprintf('  • Hội tụ chậm hơn ILS\n');
fprintf('  • Khám phá không gian rộng hơn\n');
fprintf('  • Tránh local optimum tốt hơn nhờ exploration\n');

fprintf('\n IGWO (DLH):\n');
fprintf('  • Học từ hàng xóm (neighbor wolves)\n');
fprintf('  • Greedy selection → chọn bước di chuyển tốt hơn\n');
fprintf('  • Cân bằng exploration-exploitation tốt hơn\n');

fprintf('\n Chaotic GWO:\n');
fprintf('  • Dùng Logistic Map thay vì random\n');
fprintf('  • Bước nhảy phi tuyến mạnh hơn\n');
fprintf('  • Thoát local optimum hiệu quả nhất\n');

fprintf('\n=== DONE ===\n');
