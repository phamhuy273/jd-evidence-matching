package com.matching.dto.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EvaluateAsyncRequest {

    @NotNull(message = "job_id không được để trống")
    @JsonProperty("job_id")
    private UUID jobId;

    @NotBlank(message = "Tên ứng viên không được để trống")
    @JsonProperty("candidate_name")
    private String candidateName;

    @NotBlank(message = "Đường dẫn GitHub repository không được để trống")
    @JsonProperty("github_repo_url")
    private String githubRepoUrl;

    @NotBlank(message = "commit_hash không được để trống")
    @JsonProperty("commit_hash")
    private String commitHash;
}
