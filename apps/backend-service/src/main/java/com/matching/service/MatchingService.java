package com.matching.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import com.matching.client.AiEngineClient;
import com.matching.dto.request.EvaluateAsyncRequest;
import com.matching.dto.response.ApiResponse;
import com.matching.dto.response.EvaluateAsyncResponse;
import com.matching.dto.response.JobResultsResponse;
import com.matching.dto.response.TaskStatusResponse;
import com.matching.repository.RepositoryEntityRepository;

import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class MatchingService {

    private final AiEngineClient aiEngineClient;
    private final RepositoryEntityRepository repositoryEntityRepository;

    public ApiResponse<EvaluateAsyncResponse> evaluateAsync(EvaluateAsyncRequest request) {
        log.info("Khởi tạo tác vụ đối sánh bất đồng bộ cho ứng viên: {}", request.getCandidateName());

        // Kiểm tra xem commit_hash đã tồn tại trong CSDL chưa (Caching Cấp 1 theo Mục 1.2.1)
        boolean isCached = repositoryEntityRepository
                .findByRepoUrlAndCommitHash(request.getGithubRepoUrl(), request.getCommitHash())
                .isPresent();

        if (isCached) {
            log.info("Phát hiện Cache Hit cho commit_hash: {}", request.getCommitHash());
        }

        // Gọi sang FastAPI để kích hoạt tác vụ ngầm BackgroundTasks
        return aiEngineClient.evaluateAsync(request).block();
    }

    public ApiResponse<TaskStatusResponse> getTaskStatus(String taskId) {
        return aiEngineClient.getTaskStatus(taskId).block();
    }

    public ApiResponse<JobResultsResponse> getJobResults(UUID jobId) {
        return aiEngineClient.getJobResults(jobId).block();
    }
}
