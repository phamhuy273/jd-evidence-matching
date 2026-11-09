package com.matching.client;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import com.matching.dto.request.EvaluateAsyncRequest;
import com.matching.dto.request.ExtractSkillsRequest;
import com.matching.dto.response.ApiResponse;
import com.matching.dto.response.EvaluateAsyncResponse;
import com.matching.dto.response.ExtractSkillsResponse;
import com.matching.dto.response.JobResultsResponse;
import com.matching.dto.response.TaskStatusResponse;
import com.matching.exception.AppException;
import com.matching.exception.ErrorCode;

import java.util.UUID;

@Slf4j
@Component
@RequiredArgsConstructor
public class AiEngineClient {

    private final WebClient aiEngineWebClient;

    public Mono<ApiResponse<ExtractSkillsResponse>> extractSkills(ExtractSkillsRequest request) {
        log.info("Điều phối gọi FastAPI: POST /api/v1/jobs/extract-skills cho tiêu đề: {}", request.getJobTitle());
        return aiEngineWebClient.post()
                .uri("/api/v1/jobs/extract-skills")
                .bodyValue(request)
                .retrieve()
                .onStatus(HttpStatusCode::isError, response -> {
                    log.error("FastAPI trả về mã lỗi HTTP {}", response.statusCode());
                    return Mono.error(new AppException(ErrorCode.INTERNAL_AI_ERROR, "Lỗi từ dịch vụ AI Engine"));
                })
                .bodyToMono(new ParameterizedTypeReference<ApiResponse<ExtractSkillsResponse>>() {});
    }

    public Mono<ApiResponse<EvaluateAsyncResponse>> evaluateAsync(EvaluateAsyncRequest request) {
        log.info("Điều phối gọi FastAPI: POST /api/v1/matching/evaluate-async cho repo: {}", request.getGithubRepoUrl());
        return aiEngineWebClient.post()
                .uri("/api/v1/matching/evaluate-async")
                .bodyValue(request)
                .retrieve()
                .onStatus(HttpStatusCode::isError, response -> {
                    log.error("FastAPI trả về mã lỗi HTTP {}", response.statusCode());
                    return Mono.error(new AppException(ErrorCode.GIT_CLONE_FAILED, "Không thể bắt đầu phân tích mã nguồn qua AI Engine"));
                })
                .bodyToMono(new ParameterizedTypeReference<ApiResponse<EvaluateAsyncResponse>>() {});
    }

    public Mono<ApiResponse<TaskStatusResponse>> getTaskStatus(String taskId) {
        log.info("Điều phối gọi FastAPI: GET /api/v1/matching/tasks/{}/status", taskId);
        return aiEngineWebClient.get()
                .uri("/api/v1/matching/tasks/{taskId}/status", taskId)
                .retrieve()
                .onStatus(HttpStatusCode::isError, response -> {
                    log.error("FastAPI trả về mã lỗi HTTP {}", response.statusCode());
                    return Mono.error(new AppException(ErrorCode.RESOURCE_NOT_FOUND, "Không tìm thấy task_id: " + taskId));
                })
                .bodyToMono(new ParameterizedTypeReference<ApiResponse<TaskStatusResponse>>() {});
    }

    public Mono<ApiResponse<JobResultsResponse>> getJobResults(UUID jobId) {
        log.info("Điều phối gọi FastAPI: GET /api/v1/matching/jobs/{}/results", jobId);
        return aiEngineWebClient.get()
                .uri("/api/v1/matching/jobs/{jobId}/results", jobId)
                .retrieve()
                .onStatus(HttpStatusCode::isError, response -> {
                    log.error("FastAPI trả về mã lỗi HTTP {}", response.statusCode());
                    return Mono.error(new AppException(ErrorCode.RESOURCE_NOT_FOUND, "Không tìm thấy kết quả cho job_id: " + jobId));
                })
                .bodyToMono(new ParameterizedTypeReference<ApiResponse<JobResultsResponse>>() {});
    }
}
