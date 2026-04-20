# Product Mission

## Problem

German text is often too complex for people with cognitive disabilities. Official documents, websites, and everyday content use long sentences, passive voice, foreign words, and complex grammatical structures that create barriers to understanding. "Leichte Sprache" (Simple Language) is a standardized approach to making German text accessible, but manually checking and rewriting text is time-consuming and requires specialized knowledge.

## Target Users

Content creators and editors who write or edit German-language content and need to make it accessible — including government agencies, NGOs, publishers, and web editors who are required or motivated to provide Leichte Sprache versions of their content.

## Solution

A hybrid NLP + LLM approach that combines rule-based analysis with AI-powered generation:

- **Comprehensive rule-based analysis**: 18 specialized NLP rules (powered by spaCy and BERT models) covering syntax, lexical, stylistic, and technical aspects of Leichte Sprache — providing precise, explainable violation detection rather than black-box scoring.
- **Automatic text transformation**: LLM-powered generation that doesn't just flag problems but rewrites text into compliant Leichte Sprache, supporting multiple providers (OpenAI, Anthropic, Mistral, Ollama).

This dual approach gives users both detailed diagnostics and actionable output — the most thorough automated Leichte Sprache checker and generator available.
