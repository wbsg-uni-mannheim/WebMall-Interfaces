// Results loader for Interfaces comparison website

async function loadCSV(url) {
    const response = await fetch(url);
    const text = await response.text();
    return parseCSV(text);
}

function parseCSV(text) {
    const lines = text.trim().split("\n");
    const headers = lines[0].split(",");
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        // Handle quoted values with commas
        const line = lines[i];
        const values = [];
        let current = "";
        let inQuotes = false;

        for (let j = 0; j < line.length; j++) {
            const char = line[j];
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === "," && !inQuotes) {
                values.push(current.trim());
                current = "";
            } else {
                current += char;
            }
        }
        values.push(current.trim());

        const row = {};
        headers.forEach((header, index) => {
            row[header.trim()] = values[index] || "";
        });
        data.push(row);
    }

    return { headers, data };
}

function formatNumber(value) {
    // Convert to number first
    const num = parseFloat(value);
    if (!isNaN(num)) {
        // If it's a value between 0 and 1 (inclusive), treat as percentage
        if (num >= 0 && num <= 1) {
            return (num * 100).toFixed(0) + "%";
        }
        // Otherwise return as decimal
        return num.toFixed(2);
    }

    // Handle comma-separated numbers as fallback
    if (value && value.includes(",")) {
        return value.replace(/,/g, ",");
    }
    return value;
}

function formatTokenCount(value) {
    // Remove quotes if present
    const cleanValue = value.replace(/"/g, "");

    // If it already has commas, return as is
    if (cleanValue.includes(",")) {
        return cleanValue;
    }

    // Parse as number and add commas
    const num = parseInt(cleanValue);
    if (!isNaN(num)) {
        return num.toLocaleString();
    }

    return cleanValue;
}

function calculateCost(promptTokens, completionTokens, model) {
    // Pricing per million tokens (MTok)
    const pricing = {
        "GPT-4.1": { input: 2.0, output: 8.0 },
        "Claude Sonnet 4": { input: 3.0, output: 15.0 },
    };

    if (!pricing[model]) return 0;

    const inputCost = (promptTokens / 1000000) * pricing[model].input;
    const outputCost = (completionTokens / 1000000) * pricing[model].output;

    return inputCost + outputCost;
}

function createCostTable(
    data,
    InterfaceNames,
    taskType,
    completionRates,
    runtimeData
) {
    const table = document.createElement("table");
    table.className = "table is-striped is-hoverable is-fullwidth";

    // Create header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    const headers = [
        "Interface",
        "Model",
        "Completion Rate",
        "Avg. Input Tokens",
        "Avg. Output Tokens",
        "Avg. Runtime (s)",
        "Avg. Cost ($)",
    ];
    headers.forEach((header) => {
        const th = document.createElement("th");
        th.textContent = header;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Process data - filter by task type
    const filteredData = data.data.filter(
        (row) =>
            (InterfaceNames.includes(row.Interface) ||
                row.Interface === "AX+Mem") &&
            row.Task_type === taskType
    );

    // Group by Interface and model
    const groupedData = {};
    filteredData.forEach((row) => {
        const key = `${row.Interface}-${row.Model}`;
        if (!groupedData[key]) {
            groupedData[key] = {
                interface: row.Interface,
                model: row.Model,
                taskType: row.Task_type,
                promptTokens: 0,
                completionTokens: 0,
                count: 0,
            };
        }

        const promptTokens = parseInt(
            row.prompt_tokens?.replace(/[",]/g, "") || 0
        );
        const completionTokens = parseInt(
            row.completion_tokens?.replace(/[",]/g, "") || 0
        );

        groupedData[key].promptTokens += promptTokens;
        groupedData[key].completionTokens += completionTokens;
        groupedData[key].count += 1;
    });

    // Convert to array and add completion rates and estimated runtime
    const costData = Object.values(groupedData).map((group) => {
        // Define task counts per category
        const taskCounts = {
            Basic: 48,
            Advanced: 43,
        };

        // Get the correct divisor for this task type
        const taskCount = taskCounts[taskType] || 1;

        // Calculate true averages per task (not per interface run)
        const avgPromptTokens = Math.round(group.promptTokens / taskCount);
        const avgCompletionTokens = Math.round(
            group.completionTokens / taskCount
        );
        const cost = calculateCost(
            avgPromptTokens,
            avgCompletionTokens,
            group.model
        );

        // Get completion rate from the completion rates data
        const completionRateRow = completionRates.find(
            (r) =>
                r.Interface === group.interface &&
                r.Model === group.model &&
                r.Task_type === group.taskType
        );
        const completionRate = completionRateRow
            ? parseFloat(completionRateRow.task_completion_rate)
            : 0;

        // Runtime data - lookup from loaded CSV
        let runtimeValue = "?"; // Default to unknown

        // Find runtime data for this interface, model, and task type
        if (runtimeData && runtimeData.data) {
            const runtimeRow = runtimeData.data.find(
                (row) =>
                    row.Interface === group.interface &&
                    row.Model === group.model &&
                    row.Task_type === group.taskType
            );

            if (runtimeRow && runtimeRow.Runtime_seconds) {
                runtimeValue = runtimeRow.Runtime_seconds;
            }
        }

        return {
            ...group,
            avgPromptTokens,
            avgCompletionTokens,
            cost,
            completionRate,
            runtime: runtimeValue,
        };
    });

    // Sort by completion rate (descending)
    costData.sort((a, b) => b.completionRate - a.completionRate);

    // Create body
    const tbody = document.createElement("tbody");

    costData.forEach((row) => {
        const tr = document.createElement("tr");

        // Interface
        const tdInterface = document.createElement("td");
        const nameMap = {
            RAG_Agent: "RAG Agent",
            API_MCP: "MCP Agent",
            NlWeb_elastic: "NLWeb Agent",
            "AX+Mem": "HTML Agent (Browser)",
        };
        tdInterface.textContent = nameMap[row.interface] || row.interface;
        tr.appendChild(tdInterface);

        // Model
        const tdModel = document.createElement("td");
        tdModel.textContent = row.model;
        tr.appendChild(tdModel);

        // Completion Rate
        const tdCompletionRate = document.createElement("td");
        tdCompletionRate.textContent =
            (row.completionRate * 100).toFixed(0) + "%";
        tdCompletionRate.setAttribute("data-type", "number");
        tr.appendChild(tdCompletionRate);

        // Avg Input Tokens
        const tdInputTokens = document.createElement("td");
        tdInputTokens.textContent = row.avgPromptTokens.toLocaleString();
        tdInputTokens.setAttribute("data-type", "number");
        tr.appendChild(tdInputTokens);

        // Avg Output Tokens
        const tdOutputTokens = document.createElement("td");
        tdOutputTokens.textContent = row.avgCompletionTokens.toLocaleString();
        tdOutputTokens.setAttribute("data-type", "number");
        tr.appendChild(tdOutputTokens);

        // Avg Runtime
        const tdRuntime = document.createElement("td");
        tdRuntime.textContent = row.runtime;
        if (row.runtime !== "?") {
            tdRuntime.setAttribute("data-type", "number");
        }
        tr.appendChild(tdRuntime);

        // Avg Cost
        const tdCost = document.createElement("td");
        tdCost.textContent = "$" + row.cost.toFixed(2);
        tdCost.setAttribute("data-type", "number");
        tr.appendChild(tdCost);

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);

    const tableWrapper = document.createElement("div");
    tableWrapper.className = "results-table";
    tableWrapper.appendChild(table);

    // Highlight best results after table is created
    const costColumns = [
        "Interface",
        "Model",
        "Task Type",
        "Prompt Tokens",
        "Completion Tokens",
        "Total Cost",
    ];
    highlightBestResults(table, costColumns);

    return tableWrapper;
}

function createResultsTable(data, filterFunc, InterfaceNames, taskType) {
    const table = document.createElement("table");
    table.className = "table is-striped is-hoverable is-fullwidth";

    // Filter data for our Interfaces
    const filteredData = data.data.filter(filterFunc);

    // Add AX+MEM results for comparison - only for the matching task type
    const axMemData = data.data.filter(
        (row) => row.Interface === "AX+Mem" && row.Task_type === taskType
    );

    // Combine filtered data
    const allData = [...filteredData, ...axMemData];

    // Sort by completion rate (descending), then by Interface/Interface, then Model
    allData.sort((a, b) => {
        // First sort by completion rate (descending)
        const rateA = parseFloat(a.task_completion_rate) || 0;
        const rateB = parseFloat(b.task_completion_rate) || 0;
        if (rateA !== rateB) return rateB - rateA;

        // Then by Interface/Interface
        const InterfaceOrder = {
            RAG_Agent: 1,
            API_MCP: 2,
            NlWeb_elastic: 3,
            "AX+Mem": 4,
        };
        const InterfaceA = InterfaceOrder[a.Interface] || 99;
        const InterfaceB = InterfaceOrder[b.Interface] || 99;
        if (InterfaceA !== InterfaceB) return InterfaceA - InterfaceB;

        // Finally by Model
        return (a.Model || "").localeCompare(b.Model || "");
    });

    // Create header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    // Define which columns to show
    const columnsToShow = [
        "Interface",
        "Model",
        "Task_type",
        "task_completion_rate",
        "avg_precision",
        "avg_recall",
        "f1_score",
        "prompt_tokens",
        "completion_tokens",
    ];

    const columnNames = {
        Interface: "Interface",
        Model: "Model",
        Task_type: "Task Type",
        task_completion_rate: "Completion Rate",
        avg_precision: "Precision",
        avg_recall: "Recall",
        f1_score: "F1 Score",
        prompt_tokens: "Prompt Tokens",
        completion_tokens: "Completion Tokens",
    };

    columnsToShow.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = columnNames[col] || col;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body
    const tbody = document.createElement("tbody");

    allData.forEach((row) => {
        const tr = document.createElement("tr");

        // Highlight our Interfaces
        if (InterfaceNames.some((name) => row.Interface === name)) {
            tr.style.fontWeight = "500";
        }

        columnsToShow.forEach((col) => {
            const td = document.createElement("td");
            let value = row[col];

            // Replace interface names with friendly names
            if (col === "Interface") {
                const nameMap = {
                    RAG_Agent: "RAG Agent",
                    API_MCP: "MCP Agent",
                    NlWeb_elastic: "NLWeb Agent",
                    "AX+Mem": "HTML Agent (Browser)",
                };
                value = nameMap[value] || value;
            }

            // Format numeric values
            if (
                [
                    "task_completion_rate",
                    "avg_precision",
                    "avg_recall",
                    "f1_score",
                ].includes(col)
            ) {
                value = formatNumber(value);
            }

            // Format token counts with commas
            if (["prompt_tokens", "completion_tokens"].includes(col)) {
                value = formatTokenCount(value);
            }

            td.textContent = value;

            // Add numeric class for styling
            if (
                [
                    "task_completion_rate",
                    "avg_precision",
                    "avg_recall",
                    "f1_score",
                    "prompt_tokens",
                    "completion_tokens",
                ].includes(col)
            ) {
                td.setAttribute("data-type", "number");
            }

            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);

    // Highlight best results after table is created
    const resultsColumns = [
        "Interface",
        "Model",
        "Task_type",
        "task_completion_rate",
        "avg_precision",
        "avg_recall",
        "f1_score",
        "prompt_tokens",
        "completion_tokens",
    ];
    highlightBestResults(table, resultsColumns);

    return table;
}

async function loadResults() {
    try {
        // Load runtime data
        const runtimeData = await loadCSV("./results/runtime_data.csv");

        // Load by type results
        const byTypeData = await loadCSV(
            "./results/WebMall 1.0 Results - By_Type_view.csv"
        );

        // Our Interface names in the CSV
        const InterfaceNames = ["RAG_Agent", "API_MCP", "NlWeb_elastic"];

        // Create tables for different task types
        const taskTypes = ["Basic", "Advanced"];

        taskTypes.forEach((taskType) => {
            const container = document.getElementById("results-by-type");

            // Define task counts
            const taskCounts = {
                Basic: 48,
                Advanced: 43,
            };

            // Add task type header with count
            const header = document.createElement("h4");
            header.className = "title is-5";
            header.textContent = `${taskType} Tasks (${taskCounts[taskType]} tasks)`;
            container.appendChild(header);

            // Create and append table
            const table = createResultsTable(
                byTypeData,
                (row) =>
                    InterfaceNames.includes(row.Interface) &&
                    row.Task_type === taskType,
                InterfaceNames,
                taskType
            );

            const tableWrapper = document.createElement("div");
            tableWrapper.className = "results-table";
            tableWrapper.appendChild(table);
            container.appendChild(tableWrapper);
        });

        // Load by category results
        const byCategoryData = await loadCSV(
            "./results/WebMall 1.0 Results - By_category_view.csv"
        );

        // Create detailed tables for categories
        const categoryContainer = document.getElementById(
            "results-by-category"
        );

        // Create separate tables for Basic and Advanced tasks
        const categoryTaskTypes = ["Basic", "Advanced"];

        categoryTaskTypes.forEach((taskType) => {
            // Define task counts
            const taskCounts = {
                Basic: 48,
                Advanced: 43,
            };

            // Add task type header with count
            const header = document.createElement("h4");
            header.className = "title is-5";
            header.textContent = `${taskType} Task Categories (${taskCounts[taskType]} tasks)`;
            categoryContainer.appendChild(header);

            // Create and append table
            const table = createCategoryResultsTable(
                byCategoryData,
                InterfaceNames,
                taskType
            );
            categoryContainer.appendChild(table);
        });

        // Create cost analysis tables (split by Basic and Advanced)
        const costContainer = document.getElementById("cost-analysis");

        const costTaskTypes = ["Basic", "Advanced"];

        costTaskTypes.forEach((taskType) => {
            // Define task counts
            const taskCounts = {
                Basic: 48,
                Advanced: 43,
            };

            // Add task type header with count
            const header = document.createElement("h4");
            header.className = "title is-5";
            header.textContent = `${taskType} Tasks - Cost & Performance (${taskCounts[taskType]} tasks)`;
            costContainer.appendChild(header);

            // Create and append table
            const table = createCostTable(
                byTypeData,
                InterfaceNames,
                taskType,
                byTypeData.data, // Pass the data for completion rates lookup
                runtimeData // Pass the runtime data
            );
            costContainer.appendChild(table);
        });
    } catch (error) {
        console.error("Error loading results:", error);
    }
}

function createCategoryResultsTable(data, InterfaceNames, filterTaskType) {
    const table = document.createElement("table");
    table.className = "table is-striped is-hoverable is-fullwidth";

    // Get category field name
    const categoryField =
        data.headers.find((h) => h.toLowerCase() === "category") || "category";

    // Filter data for our Interfaces
    const filteredData = data.data.filter((row) =>
        InterfaceNames.includes(row.Interface)
    );

    // Add AX+MEM results for comparison
    const axMemData = data.data.filter((row) => row.Interface === "AX+Mem");

    // Combine filtered data
    const allData = [...filteredData, ...axMemData];

    // Derive task type from category or use existing type field
    allData.forEach((row) => {
        // Use existing type field if available, otherwise derive from category
        if (row.type && row.type.trim()) {
            row.derived_type = row.type.trim();
        } else {
            const category = row[categoryField] || "";
            // Determine task type based on updated category names
            if (
                category.includes("Find Specific Product") ||
                category.includes("Find Cheapest Offer") ||
                category.includes("Best Fit Specific") ||
                category.includes("Add to Cart") ||
                category.includes("Checkout")
            ) {
                row.derived_type = "Basic";
            } else if (
                category.includes("Best Fit Vague") ||
                category.includes("Cheapest Best Fit") ||
                category.includes("Find Compatible Products") ||
                category.includes("Substitutes") ||
                category.includes("End To End") ||
                category.includes("Cheapest Best Fit Vague")
            ) {
                row.derived_type = "Advanced";
            } else {
                row.derived_type = "Other";
            }
        }
    });

    // Filter by task type if specified
    const filteredByType = filterTaskType
        ? allData.filter((row) => row.derived_type === filterTaskType)
        : allData;

    console.log(
        `Task type: ${filterTaskType}, Total rows: ${allData.length}, Filtered rows: ${filteredByType.length}`
    );

    // Sort by: Category, completion rate (descending), then Model
    filteredByType.sort((a, b) => {
        // First by Category
        const catA = a[categoryField] || "";
        const catB = b[categoryField] || "";
        if (catA !== catB) return catA.localeCompare(catB);

        // Then by completion rate (descending)
        const rateA = parseFloat(a.task_completion_rate) || 0;
        const rateB = parseFloat(b.task_completion_rate) || 0;
        if (rateA !== rateB) return rateB - rateA;

        // Finally by Model
        return (a.Model || "").localeCompare(b.Model || "");
    });

    // Create header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    // Define which columns to show (no type column since we're splitting tables)
    const columnsToShow = [
        "Interface",
        "Model",
        categoryField,
        "task_completion_rate",
        "avg_precision",
        "avg_recall",
        "f1_score",
        "prompt_tokens",
        "completion_tokens",
    ];

    const columnNames = {
        Interface: "Interface",
        Model: "Model",
        [categoryField]: "Category",
        task_completion_rate: "Completion Rate",
        avg_precision: "Precision",
        avg_recall: "Recall",
        f1_score: "F1 Score",
        prompt_tokens: "Prompt Tokens",
        completion_tokens: "Completion Tokens",
    };

    columnsToShow.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = columnNames[col] || col;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body with category separators
    const tbody = document.createElement("tbody");

    let lastCategory = null;
    
    // Helper function to get task count for a category based on task type
    const getCategoryTaskCount = (originalCategoryName, taskType) => {
        // Count tasks per category from WebMall task definitions
        // Map using actual CSV category names
        const categoryTaskCounts = {
            Basic: {
                "Find Specific Product": 12,
                "Find Cheapest Offer": 10, 
                "Best Fit Specific": 11,
                "Add to Cart": 7,
                "Checkout": 8
            },
            Advanced: {
                "Cheapest Best Fit Specific": 10,
                "Best Fit Vague": 8,
                "Cheapest Best Fit Vague": 6,
                "Substitutes": 6,
                "Find Compatible Products": 5,
                "End To End": 8
            }
        };
        
        const counts = categoryTaskCounts[taskType] || {};
        return counts[originalCategoryName] || 0;
    };

    filteredByType.forEach((row) => {
        const currentCategory = row[categoryField];

        // Add category separator row if category changed (including for first category)
        if (currentCategory !== lastCategory) {
            const separatorRow = document.createElement("tr");
            separatorRow.style.height = "30px";
            separatorRow.style.backgroundColor = "#f0f0f0";

            const separatorCell = document.createElement("td");
            separatorCell.colSpan = columnsToShow.length;
            separatorCell.style.borderTop = "2px solid #ddd";
            separatorCell.style.textAlign = "center";
            separatorCell.style.fontWeight = "bold";
            separatorCell.style.fontSize = "0.9em";
            separatorCell.style.color = "#666";

            // Format category name for display
            let categoryDisplayName = currentCategory;
            if (currentCategory) {
                categoryDisplayName = currentCategory
                    .replace("Webmall_", "")
                    .replace(/_/g, " ")
                    .replace(/([A-Z])/g, " $1")
                    .trim();

                // Apply rename mapping
                const categoryRenameMap = {
                    "Best Fit Specific":
                        "Products Fulfilling Specific Requirements",
                    "Best Fit Vague": "Satisfying Vague Requirements",
                    "Cheapest Best Fit Specific":
                        "Cheapest Offer Specific Requirements",
                    "Cheapest Best Fit Vague":
                        "Cheapest Offer Vague Requirements",
                    Substitutes: "Find Substitutes",
                };

                categoryDisplayName =
                    categoryRenameMap[categoryDisplayName] ||
                    categoryDisplayName;
            }

            // Add task count to category display name
            const taskCount = getCategoryTaskCount(currentCategory, filterTaskType);
            if (taskCount > 0) {
                categoryDisplayName += ` (${taskCount} tasks)`;
            }

            separatorCell.textContent = categoryDisplayName;

            separatorRow.appendChild(separatorCell);
            tbody.appendChild(separatorRow);
        }

        lastCategory = currentCategory;

        const tr = document.createElement("tr");

        // Highlight our Interfaces
        if (InterfaceNames.some((name) => row.Interface === name)) {
            tr.style.fontWeight = "500";
        }

        columnsToShow.forEach((col) => {
            const td = document.createElement("td");
            let value = row[col];

            // Replace interface names with friendly names
            if (col === "Interface") {
                const nameMap = {
                    RAG_Agent: "RAG Agent",
                    API_MCP: "MCP Agent",
                    NlWeb_elastic: "NLWeb Agent",
                    "AX+Mem": "HTML Agent",
                };
                value = nameMap[value] || value;
            }

            // Format category names using rename mapping
            if (col === categoryField && value) {
                // First clean the category name
                const cleanCategory = value
                    .replace("Webmall_", "")
                    .replace(/_/g, " ")
                    .replace(/([A-Z])/g, " $1")
                    .trim();

                // Apply rename mapping to match the updated CSV data
                const categoryRenameMap = {
                    "Best Fit Specific":
                        "Products Fulfilling Specific Requirements",
                    "Best Fit Vague": "Satisfying Vague Requirements",
                    "Cheapest Best Fit Specific":
                        "Cheapest Offer Specific Requirements",
                    "Cheapest Best Fit Vague":
                        "Cheapest Offer Vague Requirements",
                    Substitutes: "Find Substitutes",
                };

                value = categoryRenameMap[cleanCategory] || cleanCategory;
            }

            // Format numeric values
            if (
                [
                    "task_completion_rate",
                    "avg_precision",
                    "avg_recall",
                    "f1_score",
                ].includes(col)
            ) {
                value = formatNumber(value);
            }

            // Format token counts with commas
            if (["prompt_tokens", "completion_tokens"].includes(col)) {
                value = formatTokenCount(value);
            }

            td.textContent = value;

            // Add numeric class for styling
            if (
                [
                    "task_completion_rate",
                    "avg_precision",
                    "avg_recall",
                    "f1_score",
                    "prompt_tokens",
                    "completion_tokens",
                ].includes(col)
            ) {
                td.setAttribute("data-type", "number");
            }

            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);

    const tableWrapper = document.createElement("div");
    tableWrapper.className = "results-table";
    tableWrapper.appendChild(table);

    // Highlight best results after table is created
    highlightBestResults(table, columnsToShow);

    return tableWrapper;
}

function createCategorySummaryTable(data, InterfaceNames) {
    // Get unique categories - handle both 'Category' and 'category' field names
    const categoryField =
        data.headers.find((h) => h.toLowerCase() === "category") || "category";
    const categories = [...new Set(data.data.map((row) => row[categoryField]))]
        .filter((c) => c)
        .sort();

    // Create table
    const table = document.createElement("table");
    table.className = "table is-striped is-hoverable is-fullwidth";

    // Create header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    const th1 = document.createElement("th");
    th1.textContent = "Category";
    headerRow.appendChild(th1);

    // Add columns for each Interface
    const InterfaceDisplayNames = {
        RAG_Agent: "RAG Agent",
        API_MCP: "MCP Agent",
        NlWeb_elastic: "NLWeb Agent",
        "AX+Mem": "HTML Agent",
    };

    InterfaceNames.forEach((Interface) => {
        const th = document.createElement("th");
        th.textContent = InterfaceDisplayNames[Interface] + " (F1)";
        headerRow.appendChild(th);
    });

    // Add HTML Agent column
    const thAxMem = document.createElement("th");
    thAxMem.textContent = "HTML Agent (F1)";
    headerRow.appendChild(thAxMem);

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body
    const tbody = document.createElement("tbody");

    categories.forEach((category) => {
        const tr = document.createElement("tr");

        const tdCategory = document.createElement("td");
        // Format category name (remove 'Webmall_' prefix and replace underscores)
        const formattedCategory = category
            .replace("Webmall_", "")
            .replace(/_/g, " ")
            .replace(/([A-Z])/g, " $1")
            .trim();
        tdCategory.textContent = formattedCategory;
        tr.appendChild(tdCategory);

        // Add F1 scores for each Interface
        InterfaceNames.forEach((Interface) => {
            const td = document.createElement("td");
            td.setAttribute("data-type", "number");

            // Find best performing model for this Interface and category
            const InterfaceData = data.data.filter(
                (row) =>
                    row.Interface === Interface &&
                    row[categoryField] === category
            );

            if (InterfaceData.length > 0) {
                // Get highest F1 score
                const bestF1 = Math.max(
                    ...InterfaceData.map((row) => parseFloat(row.f1_score || 0))
                );
                td.textContent = (bestF1 * 100).toFixed(0) + "%";
            } else {
                td.textContent = "-";
            }

            tr.appendChild(td);
        });

        // Add AX+MEM data
        const tdAxMem = document.createElement("td");
        tdAxMem.setAttribute("data-type", "number");

        const axMemData = data.data.filter(
            (row) =>
                row.Interface === "AX+Mem" && row[categoryField] === category
        );

        if (axMemData.length > 0) {
            const bestF1 = Math.max(
                ...axMemData.map((row) => parseFloat(row.f1_score))
            );
            tdAxMem.textContent = (bestF1 * 100).toFixed(0) + "%";
        } else {
            tdAxMem.textContent = "-";
        }

        tr.appendChild(tdAxMem);
        tbody.appendChild(tr);
    });

    table.appendChild(tbody);

    const tableWrapper = document.createElement("div");
    tableWrapper.className = "results-table";
    tableWrapper.appendChild(table);

    // Highlight best results - for summary table, we want to highlight across all rows
    const summaryColumns = ["Category"]
        .concat(InterfaceNames)
        .concat(["HTML Agent"]);
    highlightBestInGroup(
        Array.from(table.querySelectorAll("tbody tr")),
        summaryColumns
    );

    return tableWrapper;
}

// Function to highlight best results in a table
function highlightBestResults(table, columns) {
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));

    // Skip separator rows and get data rows only
    const dataRows = rows.filter(
        (row) =>
            !row.style.backgroundColor.includes("#f0f0f0") &&
            row.cells.length > 0 &&
            row.style.height !== "10px"
    );

    if (dataRows.length === 0) return;

    // For category tables, group by category
    const isCategoryTable =
        columns.includes("Category") ||
        columns.some((col) => col.toLowerCase().includes("category"));

    if (isCategoryTable) {
        // Group rows by category
        const categoryGroups = {};
        dataRows.forEach((row) => {
            const categoryCell =
                row.cells[
                    columns.findIndex(
                        (col) =>
                            col.toLowerCase().includes("category") ||
                            col === "Category"
                    )
                ];
            if (categoryCell) {
                const category = categoryCell.textContent.trim();
                if (!categoryGroups[category]) {
                    categoryGroups[category] = [];
                }
                categoryGroups[category].push(row);
            }
        });

        // Highlight best results within each category group
        Object.values(categoryGroups).forEach((groupRows) => {
            highlightBestInGroup(groupRows, columns);
        });
    } else {
        // Highlight best results across all rows
        highlightBestInGroup(dataRows, columns);
    }
}

function highlightBestInGroup(rows, columns) {
    // Define which columns should be maximized vs minimized
    const maxColumns = [
        "task_completion_rate",
        "avg_precision",
        "avg_recall",
        "f1_score",
    ];
    const minColumns = [
        "prompt_tokens",
        "completion_tokens",
        "Prompt Tokens",
        "Completion Tokens",
        "Total Cost",
    ];

    // For each numeric column, find the best value and highlight it
    columns.forEach((column, colIndex) => {
        const isMaxColumn = maxColumns.includes(column);
        const isMinColumn = minColumns.includes(column);

        if (isMaxColumn || isMinColumn) {
            const values = [];

            rows.forEach((row, rowIndex) => {
                const cell = row.cells[colIndex];
                if (cell && cell.getAttribute("data-type") === "number") {
                    const text = cell.textContent.replace(/[%,]/g, "");
                    const value = parseFloat(text);
                    if (!isNaN(value)) {
                        values.push({ value, rowIndex, cell });
                    }
                }
            });

            if (values.length > 0) {
                // Find best value
                const bestValue = isMaxColumn
                    ? Math.max(...values.map((v) => v.value))
                    : Math.min(...values.map((v) => v.value));

                // Highlight all cells with the best value
                values.forEach(({ value, cell }) => {
                    if (value === bestValue) {
                        cell.classList.add("best-result");
                    }
                });
            }
        }
    });
}

// Load results when page is ready
document.addEventListener("DOMContentLoaded", loadResults);
