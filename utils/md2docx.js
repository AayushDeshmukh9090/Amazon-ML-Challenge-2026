// Markdown -> DOCX for Approach_Explained.md.  npm install docx ; node utils/md2docx.js Approach_Explained.md Approach_Explained.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, AlignmentType, LevelFormat, ImageRun, Footer, PageNumber,
} = require("docx");

const [, , mdPath, outPath] = process.argv;
const baseDir = path.dirname(mdPath);
let lines = fs.readFileSync(mdPath, "utf8").split(/\r?\n/);

// front matter
let title = "", subtitle = "";
if (lines[0] === "---") {
  const end = lines.indexOf("---", 1);
  for (const l of lines.slice(1, end)) {
    const m = l.match(/^(\w+):\s*"?(.*?)"?$/);
    if (m && m[1] === "title") title = m[2];
    if (m && m[1] === "subtitle") subtitle = m[2];
  }
  lines = lines.slice(end + 1);
}

const FONT = "Calibri", CODE = "Consolas";
const PAGE_W = 11906, MARGIN = 1134, CONTENT_W = PAGE_W - 2 * MARGIN;

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: CODE, size: 19, color: "7A2E0E", ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function table(rows) {
  const cells = rows.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r)).map((r) =>
    r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim()));
  const n = cells[0].length;
  const lens = Array.from({ length: n }, (_, j) => Math.max(...cells.map((r) => (r[j] || "").replace(/[*`]/g, "").length)));
  const weights = lens.map((l) => Math.max(8, Math.min(l, 60)));
  const tot = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((CONTENT_W * w) / tot));
  widths[n - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "C9D3E0" };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: cells.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: "1F3A5F", color: "auto" }
                         : (i % 2 === 0 ? { type: ShadingType.CLEAR, fill: "F3F6FA", color: "auto" } : undefined),
        borders: { top: border, bottom: border, left: border, right: border },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ children: runs(c, i === 0 ? { bold: true, color: "FFFFFF", size: 19 } : { size: 19 }) })],
      })),
    })),
  });
}

const body = [];
if (title) {
  body.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 80 },
    children: [new TextRun({ text: title, bold: true, size: 40, color: "1F3A5F", font: FONT })] }));
  if (subtitle) body.push(new Paragraph({ spacing: { after: 240 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "3B6FB6", space: 6 } },
    children: [new TextRun({ text: subtitle, italics: true, size: 24, color: "4B5563", font: FONT })] }));
}

let i = 0, para = [];
const flush = () => {
  if (para.length) prevWasNum = false;
  if (para.length) body.push(new Paragraph({ spacing: { after: 120 }, children: runs(para.join(" ")) }));
  para = [];
};
let lastList = null;   // {text, level, numbered}
let numInstance = 0, prevWasNum = false;
const pushList = () => {
  if (!lastList) return;
  if (lastList.numbered && lastList.level === 0 && !prevWasNum) numInstance++;   // new list restarts at 1
  prevWasNum = lastList.numbered || (prevWasNum && lastList.level > 0);
  const opts = lastList.numbered ? { numbering: { reference: "nums", level: lastList.level, instance: numInstance } }
                                 : { numbering: { reference: "bullets", level: lastList.level } };
  body.push(new Paragraph({ ...opts, spacing: { after: 60 }, children: runs(lastList.text) }));
  lastList = null;
};

while (i < lines.length) {
  const line = lines[i];
  if (/^\s*\|/.test(line)) {
    flush(); pushList();
    const rows = [];
    while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
    body.push(table(rows));
    body.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
    continue;
  }
  let m;
  if ((m = line.match(/^(#{1,3}) (.*)$/))) {
    flush(); pushList(); prevWasNum = false;
    const lvl = m[1].length;
    body.push(new Paragraph({ heading: lvl === 1 ? HeadingLevel.HEADING_1 : lvl === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
      children: runs(m[2]) }));
  } else if ((m = line.match(/^!\[.*\]\((.*)\)/))) {
    flush(); pushList();
    const img = fs.readFileSync(path.join(baseDir, m[1]));
    const w = 640, h = Math.round(w * 520 / 2600);
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new ImageRun({ type: "png", data: img, transformation: { width: w, height: h } })] }));
  } else if ((m = line.match(/^(\s*)- (.*)$/))) {
    flush(); pushList();
    lastList = { text: m[2], level: Math.min(2, Math.floor(m[1].length / 2)), numbered: false };
  } else if ((m = line.match(/^(\s*)\d+\. (.*)$/))) {
    flush(); pushList();
    lastList = { text: m[2], level: Math.min(2, Math.floor(m[1].length / 3)), numbered: true };
  } else if (/^\s+\S/.test(line) && lastList) {
    lastList.text += " " + line.trim();          // continuation of a list item
  } else if (line.trim() === "") {
    flush(); pushList();
  } else {
    pushList();
    para.push(line.trim());
  }
  i++;
}
flush(); pushList();

const bulletLevels = [0, 1, 2].map((l) => ({ level: l, format: LevelFormat.BULLET, text: ["•", "◦", "▪"][l],
  alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360 + l * 360, hanging: 260 } } } }));
const numLevels = [0, 1, 2].map((l) => ({ level: l, format: l === 0 ? LevelFormat.DECIMAL : LevelFormat.BULLET,
  text: l === 0 ? "%1." : "•", alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 360 + l * 360, hanging: 300 } } } }));

const doc = new Document({
  creator: "Amazon ML Challenge 2026 team",
  title: title,
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F3A5F", font: FONT },
        paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9D3E0", space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "3B6FB6", font: FONT },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: bulletLevels }, { reference: "nums", levels: numLevels }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1134, bottom: 1134, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Business Entity Resolution: approach explained · page ", size: 16, color: "6B7280" }),
                 new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "6B7280" })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outPath, b); console.log("wrote", outPath, b.length, "bytes"); });
