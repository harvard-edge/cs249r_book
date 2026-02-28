class QuartoAxeReporter {
  constructor(axeResult, options) {
    this.axeResult = axeResult;
    this.options = options;
  }

  report() {
    throw new Error("report() is an abstract method");
  }
}

class QuartoAxeJsonReporter extends QuartoAxeReporter {
  constructor(axeResult, options) {
    super(axeResult, options);
  }

  report() {
    console.log(JSON.stringify(this.axeResult, null, 2));
  }
}

class QuartoAxeConsoleReporter extends QuartoAxeReporter {
  constructor(axeResult, options) {
    super(axeResult, options);
  }

  report() {
    for (const violation of this.axeResult.violations) {
      console.log(violation.description);
      for (const node of violation.nodes) {
        for (const target of node.target) {
          console.log(target);
          console.log(document.querySelector(target));
        }
      }
    }
  }
}

class QuartoAxeDocumentReporter extends QuartoAxeReporter {
  constructor(axeResult, options) {
    super(axeResult, options);
  }

  createViolationElement(violation) {
    const violationElement = document.createElement("div");

    const descriptionElement = document.createElement("div");
    descriptionElement.className = "quarto-axe-violation-description";
    descriptionElement.innerText = `${violation.impact.replace(/^[a-z]/, match => match.toLocaleUpperCase())}: ${violation.description}`;
    violationElement.appendChild(descriptionElement);

    const helpElement = document.createElement("div");
    helpElement.className = "quarto-axe-violation-help";
    helpElement.innerText = violation.help;
    violationElement.appendChild(helpElement);

    const nodesElement = document.createElement("div");
    nodesElement.className = "quarto-axe-violation-nodes";
    violationElement.appendChild(nodesElement);
    const nodeElement = document.createElement("div");
    nodeElement.className = "quarto-axe-violation-selector";
    for (const node of violation.nodes) {
      for (const target of node.target) {
        const targetElement = document.createElement("span");
        targetElement.className = "quarto-axe-violation-target";
        targetElement.innerText = target;
        nodeElement.appendChild(targetElement);
        nodeElement.addEventListener("mouseenter", () => {
          const element = document.querySelector(target);
          if (element) {
            element.scrollIntoView({ behavior: "smooth", block: "center" });
            element.classList.add("quarto-axe-hover-highlight");
            setTimeout(() => {
              element.style.border = "";
            }, 2000);
          }
        });
        nodeElement.addEventListener("mouseleave", () => {
          const element = document.querySelector(target);
          if (element) {
            element.classList.remove("quarto-axe-hover-highlight");
          }
        });
        nodeElement.addEventListener("click", () => {
          console.log(document.querySelector(target));
        });
        nodeElement.appendChild(targetElement);
      }
      nodesElement.appendChild(nodeElement);
    }
    return violationElement;
  }

  report() {
    const violations = this.axeResult.violations;
    const reportElement = document.createElement("div");
    reportElement.className = "quarto-axe-report";
    if (violations.length === 0) {
      const noViolationsElement = document.createElement("div");
      noViolationsElement.className = "quarto-axe-no-violations";
      noViolationsElement.innerText = "No axe-core violations found.";
      reportElement.appendChild(noViolationsElement);
    }
    violations.forEach((violation) => {
      reportElement.appendChild(this.createViolationElement(violation));
    });
    document.querySelector("main").appendChild(reportElement);
  }
}

const reporters = {
  json: QuartoAxeJsonReporter,
  console: QuartoAxeConsoleReporter,
  document: QuartoAxeDocumentReporter,
};

class QuartoAxeChecker {
  constructor(opts) {
    this.options = opts;
  }
  async init() {
    const axe = (await import("https://cdn.skypack.dev/pin/axe-core@v4.10.3-aVOFXWsJaCpVrtv89pCa/mode=imports,min/optimized/axe-core.js")).default;
    const result = await axe.run({
      exclude: [
       // https://github.com/microsoft/tabster/issues/288
       // MS has claimed they won't fix this, so we need to add an exclusion to
       // all tabster elements
       "[data-tabster-dummy]"
      ],
      preload: { assets: ['cssom'], timeout: 50000 }    
    });
    const reporter = this.options === true ? new QuartoAxeConsoleReporter(result) : new reporters[this.options.output](result, this.options);
    reporter.report();
  }
}

export async function init() {
  const opts = document.querySelector("#quarto-axe-checker-options");
  if (opts) {
    const jsonOptions = JSON.parse(atob(opts.textContent));
    const checker = new QuartoAxeChecker(jsonOptions);
    await checker.init();
  }
}