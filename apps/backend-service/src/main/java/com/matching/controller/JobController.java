package com.matching.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.matching.dto.request.ExtractSkillsRequest;
import com.matching.dto.response.ApiResponse;
import com.matching.dto.response.ExtractSkillsResponse;
import com.matching.service.JobService;

@RestController
@RequestMapping("/api/v1/jobs")
@RequiredArgsConstructor
@Tag(name = "Jobs", description = "Quản lý bản mô tả công việc và bóc tách kỹ năng từ JD")
public class JobController {

    private final JobService jobService;

    @PostMapping("/extract-skills")
    @Operation(summary = "Bóc tách kỹ năng từ JD bằng LLM")
    public ResponseEntity<ApiResponse<ExtractSkillsResponse>> extractSkills(
            @Valid @RequestBody ExtractSkillsRequest request) {
        ApiResponse<ExtractSkillsResponse> response = jobService.extractAndSaveSkills(request);
        return ResponseEntity.ok(response);
    }
}
