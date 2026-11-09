package com.matching.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "repositories")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RepositoryEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @Column(name = "repo_url", nullable = false)
    private String repoUrl;

    @Column(name = "commit_hash", nullable = false, length = 40)
    private String commitHash;

    @Column(name = "default_branch", nullable = false, length = 50)
    @Builder.Default
    private String defaultBranch = "main";

    @Column(name = "has_pom", nullable = false)
    @Builder.Default
    private Boolean hasPom = false;

    @Column(name = "has_package_json", nullable = false)
    @Builder.Default
    private Boolean hasPackageJson = false;

    @Column(name = "readme_content", columnDefinition = "TEXT")
    private String readmeContent;

    @CreationTimestamp
    @Column(name = "last_parsed_at", nullable = false)
    private OffsetDateTime lastParsedAt;

    @OneToMany(mappedBy = "repository", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<CodeChunk> codeChunks = new ArrayList<>();
}
