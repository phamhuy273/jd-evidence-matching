package com.matching.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.OffsetDateTime;
import java.util.UUID;

@Entity
@Table(name = "code_chunks")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CodeChunk {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "repo_id", nullable = false)
    private RepositoryEntity repository;

    @Column(name = "file_path", nullable = false, length = 500)
    private String filePath;

    @Column(name = "class_name", length = 150)
    private String className;

    @Column(name = "method_name", length = 150)
    private String methodName;

    @Column(name = "start_line", nullable = false)
    private Integer startLine;

    @Column(name = "end_line", nullable = false)
    private Integer endLine;

    @Column(name = "context_header", nullable = false, columnDefinition = "TEXT")
    private String contextHeader;

    @Column(name = "chunk_content", nullable = false, columnDefinition = "TEXT")
    private String chunkContent;

    @Column(name = "chunk_type", nullable = false, length = 30)
    @Builder.Default
    private String chunkType = "AST_METHOD";

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private OffsetDateTime createdAt;
}
