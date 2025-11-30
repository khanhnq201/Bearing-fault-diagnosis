function downsampling_CWRU_data(input_folder, output_folder)
    % input_folder: Thư mục chứa các file .mat đầu vào
    % output_folder: Thư mục để lưu các file .mat đã downsample

    % Tạo thư mục đầu ra nếu nó chưa tồn tại
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Lấy danh sách các file .mat trong thư mục đầu vào
    mat_files = dir(fullfile(input_folder, '*.mat'));

    % Tỷ lệ downsample
    downsample_factor = 48 / 12; % Từ 48kHz xuống 12kHz, tỷ lệ là 4

    for i = 1:length(mat_files)
        file_name = mat_files(i).name;
        full_input_path = fullfile(input_folder, file_name);
        full_output_path = fullfile(output_folder, file_name);

        fprintf('Đang xử lý file: %s\n', file_name);

        % Load dữ liệu từ file .mat
        data = load(full_input_path);

        % Tạo một struct mới để lưu dữ liệu đã downsample
        downsampled_data = struct();

        % Duyệt qua tất cả các trường (key) trong dữ liệu gốc
        fields = fieldnames(data);
        for j = 1:length(fields)
            current_field = fields{j};
            field_value = data.(current_field);

            % Kiểm tra nếu trường là dữ liệu dạng số (có thể là tín hiệu)
            % và có kích thước đủ lớn để downsample
            if isnumeric(field_value) && ~isscalar(field_value) && length(field_value) > downsample_factor
                % Thực hiện downsample
                downsampled_value = resample(field_value, 1, downsample_factor);
                downsampled_data.(current_field) = downsampled_value;
            else
                % Giữ nguyên các trường không phải là dữ liệu tín hiệu
                downsampled_data.(current_field) = field_value;
            end
        end

        % Lưu dữ liệu đã downsample vào file .mat mới
        save(full_output_path, '-struct', 'downsampled_data');
        fprintf('Đã lưu file downsample: %s\n', file_name);
    end
    fprintf('Quá trình downsampling hoàn tất.\n');
end

