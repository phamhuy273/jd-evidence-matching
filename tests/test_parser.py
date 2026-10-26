"""tests/test_parser.py - Unit tests for upgraded Tree-sitter AST parser.

Governed by Task 1.3 of the SANER 2027 ERA experimental plan:
Verify that React arrow functions, function expressions, export defaults, and class component methods
are parsed into standard AST chunks without silently falling back to line-based chunking.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "ai-engine"))
from parser.tree_sitter_loader import ast_loader


class TestTreeSitterASTParser:
    """Verifies that Java and React constructs produce valid AST chunks."""

    def test_java_method_and_constructor_chunking(self):
        java_code = """package com.example.service;

public class UserService {
    private final UserRepository repository;

    public UserService(UserRepository repository) {
        this.repository = repository;
    }

    public UserDTO getUserById(Long id) {
        return repository.findById(id).map(UserDTO::from).orElseThrow();
    }
}
"""
        chunks = ast_loader.parse_and_chunk("src/main/java/UserService.java", java_code, language="java")
        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"

        chunk_types = [c["chunk_type"] for c in chunks]
        assert "AST_CONSTRUCTOR" in chunk_types
        assert "AST_METHOD" in chunk_types

        for c in chunks:
            assert c["class_name"] == "UserService"
            assert c["chunk_type"] != "LINE_FALLBACK"
            assert "// Class: UserService" in c["context_header"]

    def test_react_arrow_component_chunking(self):
        tsx_code = """import React from 'react';

interface Props {
    title: string;
}

export const Header: React.FC<Props> = ({ title }) => {
    return <header className="site-header">{title}</header>;
};
"""
        chunks = ast_loader.parse_and_chunk("src/components/Header.tsx", tsx_code, language="tsx")
        assert len(chunks) == 1
        c = chunks[0]
        assert c["chunk_type"] == "AST_ARROW_FUNCTION"
        assert c["method_name"] == "Header"
        assert "export const Header" in c["chunk_content"]
        assert "// Method: Header" in c["context_header"]

    def test_react_const_arrow_and_function_expr(self):
        ts_code = """
const calculateScore = (items: number[]): number => {
    return items.reduce((a, b) => a + b, 0);
};

const formatCurrency = function(val: number): string {
    return "$" + val.toFixed(2);
};
"""
        chunks = ast_loader.parse_and_chunk("src/utils/math.ts", ts_code, language="typescript")
        assert len(chunks) == 2
        names = [c["method_name"] for c in chunks]
        assert "calculateScore" in names
        assert "formatCurrency" in names
        assert chunks[0]["chunk_type"] == "AST_ARROW_FUNCTION"
        assert chunks[1]["chunk_type"] == "AST_FUNCTION_EXPR"

    def test_react_export_default_function(self):
        tsx_code = """import React from 'react';

export default function App() {
    return <div>Hello World</div>;
}
"""
        chunks = ast_loader.parse_and_chunk("src/App.tsx", tsx_code, language="tsx")
        assert len(chunks) == 1
        c = chunks[0]
        assert c["chunk_type"] == "AST_FUNCTION"
        assert c["method_name"] == "App"
        assert "export default function App" in c["chunk_content"]

    def test_react_export_default_anonymous_arrow(self):
        tsx_code = """import React from 'react';

export default () => {
    return <span>Anonymous Component</span>;
};
"""
        chunks = ast_loader.parse_and_chunk("src/Footer.tsx", tsx_code, language="tsx")
        assert len(chunks) == 1
        c = chunks[0]
        assert c["chunk_type"] == "AST_ARROW_FUNCTION"
        assert c["method_name"] == "default"

    def test_react_class_component_and_arrow_methods(self):
        tsx_code = """import React, { Component } from 'react';

class UserProfile extends Component {
    componentDidMount() {
        console.log("mounted");
    }

    handleSave = (e: React.FormEvent) => {
        e.preventDefault();
    };

    render() {
        return <button onClick={this.handleSave}>Save</button>;
    }
}
"""
        chunks = ast_loader.parse_and_chunk("src/UserProfile.tsx", tsx_code, language="tsx")
        assert len(chunks) == 3
        methods = {c["method_name"]: c["chunk_type"] for c in chunks}
        assert "componentDidMount" in methods and methods["componentDidMount"] == "AST_METHOD"
        assert "handleSave" in methods and methods["handleSave"] == "AST_ARROW_METHOD"
        assert "render" in methods and methods["render"] == "AST_METHOD"
        for c in chunks:
            assert c["class_name"] == "UserProfile"

    def test_syntax_error_triggers_fallback_line_chunking(self):
        broken_code = """
public class BrokenSyntax {
    public void unclosedMethod( {
        // syntax error unclosed parenthesis and braces
"""
        chunks = ast_loader.parse_and_chunk("Broken.java", broken_code, language="java")
        assert len(chunks) > 0
        assert all(c["chunk_type"] == "LINE_FALLBACK" for c in chunks)

    def test_line_chunker_window_and_step(self):
        # 120 lines of code
        lines = [f"int line_{i} = {i};" for i in range(1, 121)]
        code = "\n".join(lines)
        chunks = ast_loader.fallback_line_chunking("test.java", code, chunk_size=50, overlap=10)

        # Expected:
        # chunk 0: 1..50
        # chunk 1: 41..90
        # chunk 2: 81..120
        assert len(chunks) == 3
        assert chunks[0]["start_line"] == 1 and chunks[0]["end_line"] == 50
        assert chunks[1]["start_line"] == 41 and chunks[1]["end_line"] == 90
        assert chunks[2]["start_line"] == 81 and chunks[2]["end_line"] == 120

    def test_java_record_chunking(self):
        rec_code = """package com.example.dto;
public record OrderDTO(Long id, String name, double price) {
}"""
        chunks = ast_loader.parse_and_chunk("src/main/java/OrderDTO.java", rec_code, "java")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_RECORD"
        assert chunks[0]["class_name"] == "OrderDTO"
        assert "// Class: OrderDTO" in chunks[0]["context_header"]

    def test_java_enum_chunking(self):
        enum_code = """package com.example.model;
public enum OrderStatus {
    PENDING, PROCESSING, COMPLETED, CANCELLED
}"""
        chunks = ast_loader.parse_and_chunk("src/main/java/OrderStatus.java", enum_code, "java")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_ENUM"
        assert chunks[0]["class_name"] == "OrderStatus"

    def test_java_interface_chunking(self):
        iface_code = """package com.example.repo;
import org.springframework.data.jpa.repository.JpaRepository;
public interface CategoryRepository extends JpaRepository<Category, Long> {
}"""
        chunks = ast_loader.parse_and_chunk("src/main/java/CategoryRepository.java", iface_code, "java")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_INTERFACE"
        assert chunks[0]["class_name"] == "CategoryRepository"

    def test_java_entity_dto_chunking(self):
        entity_code = """package com.example.entity;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
@Entity
public class Product {
    @Id
    private Long id;
    private String name;
}"""
        chunks = ast_loader.parse_and_chunk("src/main/java/Product.java", entity_code, "java")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_CLASS_DECLARATION"
        assert chunks[0]["class_name"] == "Product"

    def test_react_styled_components_chunking(self):
        styled_code = """import styled from 'styled-components';
export const Container = styled.div`
  width: 100%;
  padding: 16px;
  background-color: #f5f5f5;
`;"""
        chunks = ast_loader.parse_and_chunk("src/components/Styles.ts", styled_code, "typescript")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_COMPONENT_WRAPPER"
        assert chunks[0]["method_name"] == "Container"

    def test_react_memo_and_forward_ref(self):
        memo_code = """import React, { forwardRef } from 'react';
export const FancyButton = forwardRef((props, ref) => (
  <button ref={ref} className="fancy-btn">{props.children}</button>
));"""
        chunks = ast_loader.parse_and_chunk("src/components/FancyButton.tsx", memo_code, "tsx")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_COMPONENT_WRAPPER"
        assert chunks[0]["method_name"] == "FancyButton"

    def test_redux_slice_chunking(self):
        slice_code = """import { createSlice } from '@reduxjs/toolkit';
export const authSlice = createSlice({
  name: 'auth',
  initialState: { user: null },
  reducers: {
    login: (state, action) => { state.user = action.payload; }
  }
});"""
        chunks = ast_loader.parse_and_chunk("src/store/authSlice.ts", slice_code, "typescript")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_COMPONENT_WRAPPER"
        assert chunks[0]["method_name"] == "authSlice"

    def test_typescript_types_chunking(self):
        types_code = """export interface UserProfile {
  id: string;
  name: string;
  email: string;
  roles: string[];
}"""
        chunks = ast_loader.parse_and_chunk("src/types/user.ts", types_code, "typescript")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_TYPES"

    def test_config_export_default_chunking(self):
        config_code = """import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  timeout: 30000,
});"""
        chunks = ast_loader.parse_and_chunk("playwright.config.ts", config_code, "typescript")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "AST_EXPORT_DEFAULT"

