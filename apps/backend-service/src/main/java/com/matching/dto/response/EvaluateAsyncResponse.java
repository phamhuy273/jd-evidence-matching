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
public class EvaluateAsyncResponse {

    @JsonProperty("task_id")
    private String taskId;

    private String status;

    @JsonProperty("is_cached")
    private Boolean isCached;
}
