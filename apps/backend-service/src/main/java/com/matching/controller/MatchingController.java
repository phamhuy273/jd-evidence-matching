package com.matching.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.matching.dto.request.EvaluateAsyncRequest;
import com.matching.dto.response.ApiResponse;
import com.matching.dto.response.EvaluateAsyncResponse;
import com.matching.dto.response.JobResultsResponse;
import com.matching.dto.response.TaskStatusResponse;
import com.matching.service.MatchingService;

import java.util.UUID;

@RestController
@RequestMapping("/api/v1/matching")
@RequiredArgsConstructor
@Tag(name = "Matching", description = "Điều phối đối sánh kho mã nguồn GitHub và xếp hạng ứng viên")
public class MatchingController {

    private final MatchingService matchingService;

    @PostMapping("/evaluate-async")
    @ResponseStatus(HttpStatus.ACCEPTED)
    @Operation(summary = "Kích hoạt đối sánh Repo bất đồng bộ")
    public ResponseEntity<ApiResponse<EvaluateAsyncResponse>> evaluateAsync(
            @Valid @RequestBody EvaluateAsyncRequest request) {
        ApiResponse<EvaluateAsyncResponse> response = matchingService.evaluateAsync(request);
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(response);
    }

    @GetMapping("/tasks/{task_id}/status")
    @Operation(summary = "Kiểm tra tiến trình tác vụ ngầm")
    public ResponseEntity<ApiResponse<TaskStatusResponse>> getTaskStatus(
            @PathVariable("task_id") String taskId) {
        ApiResponse<TaskStatusResponse> response = matchingService.getTaskStatus(taskId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/jobs/{job_id}/results")
    @Operation(summary = "Truy vấn danh sách xếp hạng ứng viên và minh chứng code ")
    public ResponseEntity<ApiResponse<JobResultsResponse>> getJobResults(
            @PathVariable("job_id") UUID jobId) {
        ApiResponse<JobResultsResponse> response = matchingService.getJobResults(jobId);
        return ResponseEntity.ok(response);
    }
}
