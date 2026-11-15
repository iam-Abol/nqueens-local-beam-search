const fs = require('fs');
const path = require('path');
const md = require('markdown-it')({
  html: true,
  linkify: true,
  typographer: false
});
const mk = require('markdown-it-katex');
md.use(mk);

let text = fs.readFileSync('report/report.md', 'utf8');
const figDir = path.resolve('figures');
text = text.replace(/\.\.\/figures\//g, figDir + '/');

const body = md.render(text);
const katexCss = fs.readFileSync(require.resolve('katex/dist/katex.min.css'), 'utf8');

const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Local Beam Search in N-Queens</title>
<style>
${katexCss}

@page {
  size: letter;
  margin: 0.7cm 0.9cm;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  color: #1a1e22;
  line-height: 1.26;
  font-size: 7.9pt;
  margin: 0;
  padding: 0;
}

h1 {
  font-size: 14.5pt;
  border-bottom: 1.5px solid #0366d6;
  padding-bottom: 3px;
  margin: 0 0 4px 0;
  color: #0366d6;
}

h2 {
  font-size: 11pt;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 2px;
  margin: 9px 0 3px 0;
  color: #24292e;
  page-break-after: avoid;
}

h3 {
  font-size: 9.2pt;
  margin: 7px 0 2px 0;
  color: #24292e;
  page-break-after: avoid;
}

h4 {
  font-size: 8.7pt;
  margin: 5px 0 2px 0;
  color: #444d56;
  page-break-after: avoid;
}

p, ul, ol {
  margin: 2px 0 4px 0;
}

li {
  margin-bottom: 2px;
}

table {
  border-collapse: collapse;
  width: 100%;
  margin: 5px 0;
  font-size: 6.6pt;
  page-break-inside: avoid;
}

th, td {
  border: 1px solid #dfe2e5;
  padding: 2.5px 4px;
  text-align: left;
}

th {
  background-color: #f6f8fa;
  font-weight: 600;
  color: #24292e;
}

tr:nth-child(even) {
  background-color: #fafbfc;
}

img {
  max-width: 82%;
  max-height: 135px;
  display: block;
  margin: 4px auto;
  border: 1px solid #e1e4e8;
  border-radius: 4px;
  page-break-inside: avoid;
}

code {
  background-color: rgba(27,31,35,0.05);
  border-radius: 2px;
  font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 82%;
  padding: 1px 3px;
}

hr {
  border: 0;
  height: 1px;
  background: #e1e4e8;
  margin: 6px 0;
}

.katex {
  font-size: 1.05em;
}

.katex-display {
  margin: 4px 0;
}

a {
  color: #0366d6;
  text-decoration: none;
}
</style>
</head>
<body>
${body}
</body>
</html>`;

fs.writeFileSync('report/report.html', html);
console.log('HTML rendered successfully with KaTeX!');
