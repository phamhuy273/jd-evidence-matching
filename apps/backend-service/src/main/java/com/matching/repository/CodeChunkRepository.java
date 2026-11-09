package com.matching.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.matching.entity.CodeChunk;

import java.util.List;
import java.util.UUID;

@Repository
public interface CodeChunkRepository extends JpaRepository<CodeChunk, UUID> {
    List<CodeChunk> findByRepositoryId(UUID repoId);
}
