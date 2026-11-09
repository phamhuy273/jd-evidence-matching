package com.matching.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;

/**
 * Danh mục mã lỗi nghiệp vụ quy định tại Mục 4.1 Tech Spec.
 */
@Getter
public enum ErrorCode {
    SUCCESS(200, "SUCCESS", HttpStatus.OK, "Yêu cầu được xử lý thành công"),
    ACCEPTED(202, "ACCEPTED", HttpStatus.ACCEPTED, "Tác vụ bất đồng bộ đã được tiếp nhận thành công"),
    INVALID_INPUT_DATA(400, "INVALID_INPUT_DATA", HttpStatus.BAD_REQUEST, "Dữ liệu đầu vào không hợp lệ hoặc thiếu trường bắt buộc"),
    UNAUTHORIZED(401, "UNAUTHORIZED", HttpStatus.UNAUTHORIZED, "Thiếu token xác thực hoặc phiên đăng nhập hết hạn"),
    RESOURCE_NOT_FOUND(404, "RESOURCE_NOT_FOUND", HttpStatus.NOT_FOUND, "Không tìm thấy bản ghi (JD hoặc Repo không tồn tại)"),
    GIT_CLONE_FAILED(422, "GIT_CLONE_FAILED", HttpStatus.UNPROCESSABLE_ENTITY, "Không thể truy cập hoặc clone kho mã nguồn GitHub"),
    RATE_LIMIT_EXCEEDED(429, "RATE_LIMIT_EXCEEDED", HttpStatus.TOO_MANY_REQUESTS, "Vượt quá tần suất gửi yêu cầu cho phép"),
    INTERNAL_AI_ERROR(500, "INTERNAL_AI_ERROR", HttpStatus.INTERNAL_SERVER_ERROR, "Lỗi trong quá trình nhúng vector hoặc suy luận LLM");

    private final int code;
    private final String status;
    private final HttpStatus httpStatus;
    private final String defaultMessage;

    ErrorCode(int code, String status, HttpStatus httpStatus, String defaultMessage) {
        this.code = code;
        this.status = status;
        this.httpStatus = httpStatus;
        this.defaultMessage = defaultMessage;
    }
}
