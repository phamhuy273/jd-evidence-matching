package com.matching.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("Enterprise MVP: Evidence-based JD–GitHub Matching API")
                        .version("1.0.0")
                        .description("Tài liệu đặc tả API chuẩn RESTful cho Tầng Nghiệp vụ Spring Boot điều phối hệ thống")
                        .contact(new Contact()
                                .name("Evidence Matching Team")
                                .email("contact@matching.com")));
    }
}
