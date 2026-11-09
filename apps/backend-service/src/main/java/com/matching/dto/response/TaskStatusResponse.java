package com.matching.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TaskStatusResponse {

    @JsonProperty("task_id")
    private String taskId;

    @JsonProperty("progress_status")
    private String progressStatus;

    @JsonProperty("progress_percent")
    private Integer progressPercent;
}
