// grouped tabsets

export function init() {
  window.addEventListener("pageshow", (_event) => {
    function getTabSettings() {
      const data = localStorage.getItem("quarto-persistent-tabsets-data");
      if (!data) {
        localStorage.setItem("quarto-persistent-tabsets-data", "{}");
        return {};
      }
      if (data) {
        return JSON.parse(data);
      }
    }

    function setTabSettings(data) {
      localStorage.setItem(
        "quarto-persistent-tabsets-data",
        JSON.stringify(data)
      );
    }

    function setTabState(groupName, groupValue) {
      const data = getTabSettings();
      data[groupName] = groupValue;
      setTabSettings(data);
    }

    function toggleTab(tab, active) {
      const tabPanelId = tab.getAttribute("aria-controls");
      const tabPanel = document.getElementById(tabPanelId);
      if (active) {
        tab.classList.add("active");
        tabPanel.classList.add("active");
      } else {
        tab.classList.remove("active");
        tabPanel.classList.remove("active");
      }
    }

    function toggleAll(selectedGroup, selectorsToSync) {
      for (const [thisGroup, tabs] of Object.entries(selectorsToSync)) {
        const active = selectedGroup === thisGroup;
        for (const tab of tabs) {
          toggleTab(tab, active);
        }
      }
    }

    function findSelectorsToSyncByLanguage() {
      const result = {};
      const tabs = Array.from(
        document.querySelectorAll(`div[data-group] a[id^='tabset-']`)
      );
      for (const item of tabs) {
        const div = item.parentElement.parentElement.parentElement;
        const group = div.getAttribute("data-group");
        if (!result[group]) {
          result[group] = {};
        }
        const selectorsToSync = result[group];
        const value = item.innerHTML;
        if (!selectorsToSync[value]) {
          selectorsToSync[value] = [];
        }
        selectorsToSync[value].push(item);
      }
      return result;
    }

    function setupSelectorSync() {
      const selectorsToSync = findSelectorsToSyncByLanguage();
      Object.entries(selectorsToSync).forEach(([group, tabSetsByValue]) => {
        Object.entries(tabSetsByValue).forEach(([value, items]) => {
          items.forEach((item) => {
            item.addEventListener("click", (_event) => {
              setTabState(group, value);
              toggleAll(value, selectorsToSync[group]);
            });
          });
        });
      });
      return selectorsToSync;
    }

    const selectorsToSync = setupSelectorSync();
    for (const [group, selectedName] of Object.entries(getTabSettings())) {
      const selectors = selectorsToSync[group];
      // it's possible that stale state gives us empty selections, so we explicitly check here.
      if (selectors) {
        toggleAll(selectedName, selectors);
      }
    }
  });
}
