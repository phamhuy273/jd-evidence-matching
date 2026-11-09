package com.matching.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

/**
 * StandardResponseWrapper tuân thủ Mục 4.1 Tech Spec.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ApiResponse<T> {
    private int code;
    private String status;
    private String message;
    private T data;
    @Builder.Default
    private String timestamp = Instant.now().toString();

    public static <T> ApiResponse<T> success(T data) {
        return ApiResponse.<T>builder()
                .code(200)
                .status("SUCCESS")
                .message("Thao tác thực hiện thành công.")
                .data(data)
                .timestamp(Instant.now().toString())
                .build();
    }

    public static <T> ApiResponse<T> success(String message, T data) {
        return ApiResponse.<T>builder()
                .code(200)
                .status("SUCCESS")
                .message(message)
                .data(data)
                .timestamp(Instant.now().toString())
                .build();
    }

    public static <T> ApiResponse<T> accepted(String message, T data) {
        return ApiResponse.<T>builder()
                .code(202)
                .status("ACCEPTED")
                .message(message)
                .data(data)
                .timestamp(Instant.now().toString())
                .build();
    }

    public static <T> ApiResponse<T> error(int code, String status, String message) {
        return ApiResponse.<T>builder()
                .code(code)
                .status(status)
                .message(message)
                .timestamp(Instant.now().toString())
                .build();
    }
}
