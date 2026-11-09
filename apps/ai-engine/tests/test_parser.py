import pytest
from parser.tree_sitter_loader import ASTParserLoader

@pytest.fixture
def parser_loader():
    return ASTParserLoader()

def test_ast_chunking_java(parser_loader):
    sample_java = """package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
"""
    chunks = parser_loader.parse_and_chunk("Calculator.java", sample_java, language="java")
    assert len(chunks) >= 2
    # Check that methods were extracted
    methods = [c["chunk_content"] for c in chunks]
    assert any("add" in m for m in methods)
    assert any("subtract" in m for m in methods)
    assert chunks[0]["file_path"] == "Calculator.java"
    # Check class_name and method_name extraction
    assert all(c["class_name"] == "Calculator" for c in chunks)
    method_names = [c["method_name"] for c in chunks]
    assert "add" in method_names
    assert "subtract" in method_names
    # Check Context Header contains File, Class, and Lines
    assert "// File: Calculator.java" in chunks[0]["context_header"]
    assert "// Class: Calculator" in chunks[0]["context_header"]
    assert "// Method: " in chunks[0]["context_header"]

def test_ast_chunking_javascript(parser_loader):
    sample_js = """class MathUtil {
    multiply(a, b) {
        return a * b;
    }
}

function greet(name) {
    return 'Hello, ' + name;
}
"""
    chunks = parser_loader.parse_and_chunk("utils.js", sample_js, language="javascript")
    assert len(chunks) >= 2
    methods = [c["chunk_content"] for c in chunks]
    assert any("multiply" in m for m in methods)
    assert any("greet" in m for m in methods)

    # Class method should have class_name
    class_method = next(c for c in chunks if c["method_name"] == "multiply")
    assert class_method["class_name"] == "MathUtil"
    assert "// Class: MathUtil" in class_method["context_header"]

    # Standalone function should have class_name = None
    func = next(c for c in chunks if c["method_name"] == "greet")
    assert func["class_name"] is None
    assert "// Method: greet" in func["context_header"]

def test_fallback_line_chunking_on_syntax_error(parser_loader):
    # Syntax error with missing braces
    invalid_code = "public class Broken { void test() { if (true) { " * 20
    chunks = parser_loader.parse_and_chunk("Broken.java", invalid_code, language="java")
    assert len(chunks) > 0
    # Fallback line chunking should produce chunks
    assert all(c["chunk_type"] in ("LINE_FALLBACK", "AST_METHOD") for c in chunks)
    assert all("Broken.java" in c["context_header"] for c in chunks)

def test_fallback_line_chunking_directly(parser_loader):
    lines = [f"System.out.println({i});" for i in range(120)]
    source = "\n".join(lines)
    chunks = parser_loader.fallback_line_chunking("TestLines.java", source, chunk_size=50, overlap=10)
    assert len(chunks) >= 2
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 50
    assert chunks[0]["chunk_type"] == "LINE_FALLBACK"
