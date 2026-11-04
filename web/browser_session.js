// Trusted static component. Only public routing and revision metadata cross to JS.
let active = null;

export default function(component) {
  const {data, setStateValue} = component;
  if (!active || active.revision !== data.revision) {
    active = {revision: data.revision, promise: null};
    active.promise = (async () => {
      const options = {
        method: "POST", credentials: "include", redirect: "error",
        headers: {"Content-Type": "application/json", "X-Browser-Session": "1"},
      };
      const response = await fetch(`${data.endpoint}/${data.operation}`, {
        ...options, body: JSON.stringify({revision: data.revision}),
      });
      if (!response.ok) throw new Error("session-unavailable");
      const issued = await response.json();
      // Set-Cookie is not readable in JS; confirm actual cookie return to the API.
      const confirmation = await fetch(`${data.endpoint}/check`, options);
      if (!confirmation.ok) throw new Error("session-unavailable");
      const retained = await confirmation.json();
      const installed = data.operation === "clear"
        ? retained.status === "empty"
        : retained.status === "ready" && retained.revision === issued.revision
          && issued.revision === data.revision;
      if (!installed) throw new Error("cookie-rejected");
    })();
  }
  // Shared in-flight work prevents duplicate mutation POSTs on component rerenders.
  const current = active;
  current.promise.then(() => {
    if (!current.handled) {
      current.handled = true;
      window.location.reload();
    }
  }).catch(() => {
    if (!current.handled) {
      current.handled = true;
      setStateValue("failed", true);
    }
  });
}
