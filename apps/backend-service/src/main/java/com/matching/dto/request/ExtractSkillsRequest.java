package com.matching.dto.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExtractSkillsRequest {

    @NotBlank(message = "Tiêu đề công việc không được để trống")
    @JsonProperty("job_title")
    private String jobTitle;

    @NotBlank(message = "Nội dung JD không được để trống")
    @Size(min = 10, message = "Nội dung JD quá ngắn để trích xuất (tối thiểu 10 ký tự)")
    @JsonProperty("raw_jd_text")
    private String rawJdText;
}
