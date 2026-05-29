const SYMBOLS = [
  { label: "x", insert: "x" },
  { label: "y", insert: "y" },
  { label: "+", insert: " + " },
  { label: "−", insert: " - " },
  { label: "×", insert: " * " },
  { label: "÷", insert: " / " },
  { label: "=", insert: " = " },
  { label: "(", insert: "(" },
  { label: ")", insert: ")" },
  { label: "²", insert: "^2" },
  { label: "½", insert: "1/2" },
  { label: "≤", insert: " <= " },
  { label: "≥", insert: " >= " },
  { label: "≠", insert: " != " },
  { label: "2x", insert: "2x" },
  { label: "3x", insert: "3x" },
  { label: "π", insert: "pi" },
];

const EXAMPLES = [
  {
    label: "Packages problem (Class 1)",
    text:
      "In a store, 9 packages of books are bought, which has an additional price of $75. How much does each package cost if the total paid was $1,400?",
  },
  {
    label: "Room perimeter — twice width (Class 2)",
    text:
      "The length of a room is twice as long as its width. If its perimeter is 21 m, what are its dimensions?",
  },
  {
    label: "Land perimeter — three times width",
    text:
      "The perimeter of a rectangular piece of land is 56 meters. If the length is three times the width, what are the dimensions?",
  },
  {
    label: "Equation only",
    text: "9x + 75 = 1400",
  },
  {
    label: "Equation only",
    text: "2x + 6 = 21",
  },
];

const questionInput = document.getElementById("questionInput");
const symbolGrid = document.getElementById("symbolGrid");
const exampleChips = document.getElementById("exampleChips");
const solveBtn = document.getElementById("solveBtn");
const spinner = document.getElementById("spinner");
const output = document.getElementById("output");
const sourcesBlock = document.getElementById("sourcesBlock");
const sourcesList = document.getElementById("sourcesList");
const statusPill = document.getElementById("statusPill");
const problemType = document.getElementById("problemType");

/** Convert model output like `[ x = \frac{9}{2} ]` into KaTeX-friendly `$$...$$`. */
function normalizeLatex(text) {
  let out = text;

  // Bracket-wrapped LaTeX: [ x = \frac{1325}{9} ]
  out = out.replace(/\[\s*([^\[\]]*\\[^\[\]]*)\s*\]/g, (_, expr) => `$$${expr.trim()}$$`);

  // Parenthesis-wrapped LaTeX: ( x \approx 147.22 ) when it contains a backslash
  out = out.replace(
    /\(\s*([^()]*\\[^()]*)\s*\)/g,
    (_, expr) => `$${expr.trim()}$`
  );

  return out;
}

function renderAnswerMarkdown(markdown) {
  const normalized = normalizeLatex(markdown);
  output.innerHTML = marked.parse(normalized);

  if (typeof renderMathInElement === "function") {
    renderMathInElement(output, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  }
}

function insertAtCursor(text) {
  const start = questionInput.selectionStart;
  const end = questionInput.selectionEnd;
  const before = questionInput.value.slice(0, start);
  const after = questionInput.value.slice(end);
  questionInput.value = before + text + after;
  const pos = start + text.length;
  questionInput.selectionStart = questionInput.selectionEnd = pos;
  questionInput.focus();
}

SYMBOLS.forEach(({ label, insert }) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "symbol-btn";
  btn.textContent = label;
  btn.title = `Insert ${label}`;
  btn.addEventListener("click", () => insertAtCursor(insert));
  symbolGrid.appendChild(btn);
});

EXAMPLES.forEach(({ label, text }) => {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip";
  chip.textContent = label;
  chip.addEventListener("click", () => {
    questionInput.value = text;
    questionInput.focus();
  });
  exampleChips.appendChild(chip);
});

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    statusPill.textContent = `Ready · ${data.indexed_chunks} pedagogy chunks`;
    statusPill.classList.add("ok");
  } catch {
    statusPill.textContent = "Server offline";
    statusPill.classList.add("error");
  }
}

async function solve() {
  const question = questionInput.value.trim();
  if (!question) {
    output.className = "output error-msg";
    output.textContent = "Please enter a question or equation.";
    return;
  }

  solveBtn.disabled = true;
  spinner.classList.remove("hidden");
  output.className = "output placeholder";
  output.textContent = "Thinking through the problem with paper-based pedagogy…";
  sourcesBlock.classList.add("hidden");
  problemType.classList.add("hidden");

  try {
    const res = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }

    output.className = "output markdown-body";
    renderAnswerMarkdown(data.answer);

    problemType.textContent = data.is_word_problem
      ? "Word problem (E1 focus)"
      : "Equation (E1 → E2)";
    problemType.classList.remove("hidden");

    if (data.sources?.length) {
      sourcesList.innerHTML = data.sources
        .map(
          (s) =>
            `<li><strong>${escapeHtml(s.title)}</strong> — ${escapeHtml(s.section)}</li>`
        )
        .join("");
      sourcesBlock.classList.remove("hidden");
    }
  } catch (err) {
    output.className = "output error-msg";
    output.textContent = err.message || "Something went wrong. Check your API key and try again.";
  } finally {
    solveBtn.disabled = false;
    spinner.classList.add("hidden");
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

solveBtn.addEventListener("click", solve);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) solve();
});

checkHealth();
