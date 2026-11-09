"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Module: TREE-SITTER AST PROGRESSIVE DISCLOSURE CHUNKER (UPGRADED)
Grammars: Java, TypeScript, JavaScript, TSX
Target: IEEE SANER 2027 (ERA Track - CORE A) & Double-Anonymous Peer Review
Governed by: Rules R1-R46 (Path B Factorial 2x2 Experimental Design)
=============================================================================
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ai-engine.parser")


class ASTParserLoader:
    def __init__(self):
        self.parsers = {}
        self._init_parsers()

    def _init_parsers(self):
        """
        Load syntax grammars for Java, JavaScript, TypeScript, and TSX from Tree-sitter bindings.
        """
        try:
            from tree_sitter import Language, Parser
            self.Parser = Parser

            try:
                import tree_sitter_java
                java_lang = Language(tree_sitter_java.language())
                java_parser = Parser(java_lang)
                self.parsers["java"] = java_parser
                logger.info("Successfully loaded Tree-sitter Java grammar.")
            except Exception as e:
                logger.warning(f"Could not load tree_sitter_java: {e}")

            try:
                import tree_sitter_javascript
                js_lang = Language(tree_sitter_javascript.language())
                js_parser = Parser(js_lang)
                self.parsers["javascript"] = js_parser
                logger.info("Successfully loaded Tree-sitter JavaScript grammar.")
            except Exception as e:
                logger.warning(f"Could not load tree_sitter_javascript: {e}")

            try:
                import tree_sitter_typescript
                ts_lang = Language(tree_sitter_typescript.language_typescript())
                ts_parser = Parser(ts_lang)
                self.parsers["typescript"] = ts_parser

                tsx_lang = Language(tree_sitter_typescript.language_tsx())
                self.parsers["tsx"] = Parser(tsx_lang)
                logger.info("Successfully loaded Tree-sitter TypeScript/TSX grammar.")
            except Exception as e:
                logger.warning(f"Could not load tree_sitter_typescript: {e}")

        except ImportError:
            logger.warning("Tree-sitter library is not installed in the current environment.")

    def parse_and_chunk(
        self,
        file_path: str,
        source_code: str,
        language: str = "java",
        include_export_prefix: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Segment source code at syntax-aware boundaries.
        Supports:
          - Java: method_declaration, constructor_declaration, record_declaration,
                  enum_declaration, interface_declaration, class_declaration (DTO/Entity)
          - TypeScript / JS / TSX:
              - function_declaration, method_definition, public_field_definition
              - arrow_function, function_expression
              - call_expression & tagged_template_expression (React.memo, forwardRef,
                styled-components, createSlice, createContext, defineConfig)
              - interface_declaration, type_alias_declaration
              - export default declarations and component objects
        Prepends standardized hierarchical context headers (file_path, class, method, lines).
        If zero syntactic nodes are produced or fatal parse exception occurs, shifts to fallback line slicing.
        """
        lang_key = language.lower()
        if lang_key in ("ts", "typescript"):
            parser = self.parsers.get("typescript") or self.parsers.get("tsx")
        elif lang_key in ("tsx", "jsx", "react"):
            parser = self.parsers.get("tsx") or self.parsers.get("typescript")
        elif lang_key in ("js", "javascript"):
            parser = self.parsers.get("javascript")
        else:
            parser = self.parsers.get("java")

        if not parser:
            logger.info(f"No AST parser found for '{language}', shifting to fallback line-based chunking.")
            return self.fallback_line_chunking(file_path, source_code)

        try:
            source_bytes = source_code.encode("utf8")
            tree = parser.parse(source_bytes)
            root_node = tree.root_node

            def get_node_text(n):
                if n is None:
                    return ""
                return source_bytes[n.start_byte:n.end_byte].decode("utf8", errors="replace")

            def get_identifier(n):
                if n is None:
                    return None
                name_node = n.child_by_field_name("name")
                if name_node:
                    return get_node_text(name_node)
                for child in n.children:
                    if child.type in ("identifier", "type_identifier", "property_identifier"):
                        return get_node_text(child)
                return None

            chunks = []

            def add_chunk(target_node, method_name, current_class, chunk_type):
                start_line = source_bytes[:target_node.start_byte].count(b'\n') + 1
                end_line = source_bytes[:target_node.end_byte].count(b'\n') + 1
                chunk_text = get_node_text(target_node)
                loc = end_line - start_line + 1

                header_lines = [f"// File: {file_path}"]
                cls_val = current_class or (method_name if chunk_type in ("AST_CLASS", "AST_RECORD", "AST_INTERFACE", "AST_ENUM", "AST_CLASS_DECLARATION") else None)
                if cls_val:
                    header_lines.append(f"// Class: {cls_val}")
                if method_name:
                    header_lines.append(f"// Method: {method_name}")
                header_lines.append(f"// Lines: {start_line}-{end_line}")
                context_header = "\n".join(header_lines)

                chunks.append({
                    "file_path": file_path,
                    "class_name": cls_val,
                    "method_name": method_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "loc": loc,
                    "context_header": context_header,
                    "chunk_content": chunk_text,
                    "chunk_type": chunk_type
                })

            is_java = lang_key in ("java",)

            if is_java:
                for node in root_node.children:
                    nt = node.type
                    if nt in ("class_declaration", "record_declaration", "enum_declaration", "interface_declaration"):
                        cls_name = get_identifier(node)
                        body = node.child_by_field_name("body")
                        methods = []
                        if body:
                            for child in body.children:
                                if child.type in ("method_declaration", "constructor_declaration"):
                                    methods.append(child)

                        if nt == "record_declaration":
                            add_chunk(node, cls_name, cls_name, "AST_RECORD")
                        elif nt == "enum_declaration":
                            add_chunk(node, cls_name, cls_name, "AST_ENUM")
                        elif nt == "interface_declaration":
                            s_l = source_bytes[:node.start_byte].count(b'\n') + 1
                            e_l = source_bytes[:node.end_byte].count(b'\n') + 1
                            if (e_l - s_l + 1) <= 80 or len(methods) == 0:
                                add_chunk(node, cls_name, cls_name, "AST_INTERFACE")
                            else:
                                for m in methods:
                                    m_name = get_identifier(m)
                                    add_chunk(m, m_name, cls_name, "AST_METHOD")
                        elif nt == "class_declaration":
                            if len(methods) == 0:
                                add_chunk(node, cls_name, cls_name, "AST_CLASS_DECLARATION")
                            else:
                                for m in methods:
                                    m_name = get_identifier(m)
                                    ctype = "AST_CONSTRUCTOR" if m.type == "constructor_declaration" else "AST_METHOD"
                                    add_chunk(m, m_name, cls_name, ctype)
            else:
                # TypeScript / JS / TSX
                type_decls = []

                def flush_type_decls():
                    nonlocal type_decls
                    if not type_decls:
                        return
                    first_node = type_decls[0]
                    last_node = type_decls[-1]
                    s_line = source_bytes[:first_node.start_byte].count(b'\n') + 1
                    e_line = source_bytes[:last_node.end_byte].count(b'\n') + 1
                    loc = e_line - s_line + 1
                    if loc < 5:
                        type_decls = []
                        return
                    combined_text = source_bytes[first_node.start_byte:last_node.end_byte].decode("utf8", errors="replace")
                    names = [get_identifier(n) for n in type_decls]
                    name_str = ", ".join([n for n in names if n]) or "Types"
                    header_lines = [
                        f"// File: {file_path}",
                        f"// Method: {name_str}",
                        f"// Lines: {s_line}-{e_line}"
                    ]
                    chunks.append({
                        "file_path": file_path,
                        "class_name": None,
                        "method_name": name_str,
                        "start_line": s_line,
                        "end_line": e_line,
                        "loc": loc,
                        "context_header": "\n".join(header_lines),
                        "chunk_content": combined_text,
                        "chunk_type": "AST_TYPES"
                    })
                    type_decls = []

                for node in root_node.children:
                    nt = node.type
                    target = node
                    actual = node

                    if nt == "export_statement":
                        for c in node.children:
                            if c.type in ("lexical_declaration", "variable_declaration", "function_declaration",
                                          "class_declaration", "interface_declaration", "type_alias_declaration"):
                                actual = c
                                break
                            elif c.type in ("arrow_function", "function_expression"):
                                actual = c
                                break

                    act_nt = actual.type

                    # Types and interfaces
                    if act_nt in ("interface_declaration", "type_alias_declaration"):
                        type_decls.append(target)
                        continue
                    else:
                        flush_type_decls()

                    # Standard function declaration
                    if act_nt == "function_declaration":
                        fn_name = get_identifier(actual)
                        add_chunk(target, fn_name, None, "AST_FUNCTION")

                    # Class component
                    elif act_nt == "class_declaration":
                        cls_name = get_identifier(actual)
                        body = actual.child_by_field_name("body")
                        methods = []
                        if body:
                            for child in body.children:
                                if child.type in ("method_definition", "public_field_definition", "field_definition"):
                                    m_val = child.child_by_field_name("value")
                                    if child.type == "method_definition" or (m_val and m_val.type in ("arrow_function", "function_expression")):
                                        m_name = get_identifier(child)
                                        ctype = "AST_ARROW_METHOD" if child.type != "method_definition" else "AST_METHOD"
                                        add_chunk(child, m_name, cls_name, ctype)
                                        methods.append(child)
                        if not methods:
                            add_chunk(target, cls_name, cls_name, "AST_CLASS")

                    # Const / let variable declarations
                    elif act_nt in ("lexical_declaration", "variable_declaration"):
                        for decl in actual.children:
                            if decl.type == "variable_declarator":
                                var_name = get_identifier(decl)
                                val = decl.child_by_field_name("value")
                                if val:
                                    vt = val.type
                                    if vt == "arrow_function":
                                        add_chunk(target, var_name, None, "AST_ARROW_FUNCTION")
                                    elif vt == "function_expression":
                                        add_chunk(target, var_name, None, "AST_FUNCTION_EXPR")
                                    elif vt in ("call_expression", "tagged_template_expression"):
                                        add_chunk(target, var_name, None, "AST_COMPONENT_WRAPPER")
                                    elif vt in ("array", "array_expression", "object", "object_expression"):
                                        add_chunk(target, var_name, None, "AST_CONSTANT_DECLARATION")

                    # Export default with expression / call
                    elif nt == "export_statement":
                        exp = None
                        for c in node.children:
                            if c.type in ("call_expression", "arrow_function", "function_expression", "object", "array"):
                                exp = c
                                break
                        if exp:
                            ctype = "AST_ARROW_FUNCTION" if exp.type == "arrow_function" else ("AST_FUNCTION_EXPR" if exp.type == "function_expression" else "AST_EXPORT_DEFAULT")
                            add_chunk(node, "default", None, ctype)

                    # Top-level call expressions (e.g. describe / test blocks, defineConfig)
                    elif nt == "expression_statement":
                        for c in node.children:
                            if c.type == "call_expression":
                                callee = c.child_by_field_name("function")
                                c_name = get_node_text(callee) if callee else "call"
                                add_chunk(node, c_name, None, "AST_EXPRESSION")

                flush_type_decls()

            if not chunks:
                logger.info(f"File {file_path} produced 0 AST chunks; falling back to line chunking.")
                return self.fallback_line_chunking(file_path, source_code)

            # Split oversized nodes (> 100 LOC) at syntactic statement boundaries
            final_chunks = []
            for c in chunks:
                if c["loc"] > 100:
                    lines = c["chunk_content"].splitlines()
                    total_l = len(lines)
                    step = 50
                    for s_idx in range(0, total_l, step):
                        e_idx = min(s_idx + 70, total_l)
                        sub_lines = lines[s_idx:e_idx]
                        sub_start = c["start_line"] + s_idx
                        sub_end = c["start_line"] + e_idx - 1
                        sub_header = f"// File: {c['file_path']}"
                        if c.get("class_name"):
                            sub_header += f"\n// Class: {c['class_name']}"
                        if c.get("method_name"):
                            sub_header += f"\n// Method: {c['method_name']}"
                        sub_header += f"\n// Lines: {sub_start}-{sub_end}"
                        final_chunks.append({
                            "file_path": c["file_path"],
                            "class_name": c.get("class_name"),
                            "method_name": c.get("method_name"),
                            "start_line": sub_start,
                            "end_line": sub_end,
                            "loc": len(sub_lines),
                            "context_header": sub_header,
                            "chunk_content": "\n".join(sub_lines),
                            "chunk_type": c["chunk_type"]
                        })
                        if e_idx == total_l:
                            break
                else:
                    final_chunks.append(c)

            return final_chunks

        except Exception as e:
            logger.warning(f"AST chunking exception for {file_path}, falling back: {e}")
            return self.fallback_line_chunking(file_path, source_code)

    def fallback_line_chunking(
        self,
        file_path: str,
        source_code: str,
        chunk_size: int = 50,
        overlap: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fallback line-based chunker:
        Segments code into fixed sliding windows (default 50 LOC, 10 overlap, step 40).
        """
        lines = source_code.splitlines()
        chunks = []
        total_lines = len(lines)
        if total_lines == 0:
            return chunks

        step = max(1, chunk_size - overlap)
        for start_idx in range(0, total_lines, step):
            end_idx = min(start_idx + chunk_size, total_lines)
            chunk_lines = lines[start_idx:end_idx]
            context_header = f"// File: {file_path}\n// Lines: {start_idx + 1}-{end_idx}"
            chunk_text = "\n".join(chunk_lines)

            chunks.append({
                "file_path": file_path,
                "class_name": None,
                "method_name": None,
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "loc": len(chunk_lines),
                "context_header": context_header,
                "chunk_content": chunk_text,
                "chunk_type": "LINE_FALLBACK"
            })
            if end_idx == total_lines:
                break
        return chunks


ast_loader = ASTParserLoader()
