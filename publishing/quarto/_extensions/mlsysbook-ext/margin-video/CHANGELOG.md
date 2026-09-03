# Changelog

All notable changes to the Margin Video extension will be documented in this file.

## [1.0.0] - 2024-12-08

### Added
- Initial release of margin-video extension
- YouTube video embedding as margin notes
- Automatic video numbering in HTML output
- QR code generation for PDF output
- Format-specific rendering (HTML vs PDF)
- YouTube URL validation with clear error messages
- Support for multiple YouTube URL formats:
  - `youtube.com/watch?v=ID`
  - `youtu.be/ID`
  - `youtube.com/embed/ID`
- Configuration options via kwargs:
  - `aspect-ratio`: Custom video aspect ratio (default: "16/9")
  - `start`: Start time in seconds
  - `autoplay`: Enable autoplay (default: false)
- Comprehensive documentation and examples
- Error handling for missing or invalid arguments

### Features
- **HTML Output**: Responsive iframe with configurable aspect ratio
- **PDF Output**: QR code with margin note and clickable link
- **Validation**: YouTube-only support with helpful error messages
- **Flexibility**: Configurable video parameters and styling
- **Documentation**: Complete README with usage examples

### Technical Details
- Quarto >= 1.2.0 required
- MIT licensed
- Self-contained extension with no external dependencies
