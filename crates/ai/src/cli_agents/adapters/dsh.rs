//! DeepSeek Harness (DSH) in its headless profile (`dsh --profile headless --json`).
//!
//! The desktop app bundles the CLI that drives its runtime (`…/resources/runtime/cli/bin/dsh.cmd`
//! on Windows); when it isn't on PATH, its full path can be set as the backend's command. One
//! task per run: the answer is the `final` event's text, progress arrives as `status` events,
//! and a failed turn names why in its `turn_end` reason. Every run reports the session it used,
//! and the next message continues it with `--session-id`.
//!
//! The model is picked by the profile's own configuration layers (`cordis.patch.yml`), not a
//! flag, so `cli_model` doesn't reach it; a missing API key is named by the run itself
//! (`MISSING_CREDENTIAL`). The events of a turn that uses tools haven't been observed yet
//! (R2.2): unknown events are ignored until then, so such a run still works, its progress
//! unheard. Nothing here asks us before a tool runs, so a Clean room build's reads aren't
//! checked — the same gap as every agent that never asks us, left to the approval decision
//! (R2.2).

use super::*;

pub struct Dsh;

impl AgentAdapter for Dsh {
    fn ids(&self) -> &'static [&'static str] {
        &["dsh", "deepseek-harness"]
    }

    fn args(&self, cx: &ArgsCx) -> Vec<String> {
        let mut a = vec!["--profile".into(), "headless".into(), "--json".into()];
        if let Some(r) = cx.resume {
            a.extend(["--session-id".into(), r.to_string()]);
        }
        a.extend(cx.options.args.iter().cloned());
        a.push(format!("{}{}", cx.preamble, cx.prompt));
        a
    }

    fn required_flags(&self) -> (&'static [&'static str], &'static [&'static str]) {
        (&["--help"], &["--profile", "headless"])
    }

    /// Plain `dsh <task>` boots a profile's own interface; a failed flag check keeps this
    /// adapter, so the run says what's missing instead of opening one.
    fn plain_fallback(&self) -> bool {
        false
    }

    fn resumes(&self) -> bool {
        true
    }

    /// The one answer is the `final` event's text; a `turn_end` that failed names why. The
    /// `session` event is what makes the next message a continuation.
    fn parse(&self, p: &mut StreamParser, v: &Value) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        match v.get("type").and_then(Value::as_str).unwrap_or_default() {
            "session" => {
                if let Some(id) = v.get("sessionId").and_then(Value::as_str) {
                    p.session_id = Some(id.to_string());
                }
            }
            "status" => {
                let failed = v.get("phase").and_then(Value::as_str) == Some("turn_end") && v.pointer("/reason/kind").and_then(Value::as_str) == Some("error");
                if failed {
                    let m = v.pointer("/reason/error/message").and_then(Value::as_str).unwrap_or("unknown error");
                    out.push(AgentEvent::Error { message: format!("DeepSeek Harness: {m}"), remediation: None, log: None });
                }
            }
            "final" => {
                p.saw_result = true;
                let text = v.get("text").and_then(Value::as_str).unwrap_or_default();
                if !text.is_empty() {
                    out.push(AgentEvent::TextDelta { text: text.to_string() });
                }
            }
            _ => {}
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_with(resume: Option<&str>, options: &BackendOptions) -> Vec<String> {
        let mcp = McpCommand { command: "craft-anything-cli".into(), args: vec!["mcp".into()], notes: None };
        let cx = ArgsCx { options, prompt: "make a game", preamble: "PRE ", mcp: &mcp, resume, builder: false, reads_checked: false, question: false };
        Dsh.args(&cx)
    }

    #[test]
    fn one_task_per_run_after_the_profile_and_json_flags() {
        assert_eq!(args_with(None, &BackendOptions::default()), ["--profile", "headless", "--json", "PRE make a game"]);
        let options = BackendOptions { args: vec!["--patch".into(), "extra.yml".into()], ..Default::default() };
        assert_eq!(
            args_with(Some("session-1"), &options),
            ["--profile", "headless", "--json", "--session-id", "session-1", "--patch", "extra.yml", "PRE make a game"]
        );
    }

    #[test]
    fn dsh_is_registered_and_kept_on_a_failed_flag_check() {
        assert!(AgentAdapters::built_in().for_agent("dsh").ids().contains(&"dsh"), "registered, not the plain adapter");
        assert!(AgentAdapters::built_in().for_agent("deepseek-harness").ids().contains(&"deepseek-harness"));
        assert_eq!(Dsh.required_flags(), (&["--help"][..], &["--profile", "headless"][..]));
        assert!(!Dsh.plain_fallback() && Dsh.resumes() && Dsh.streams_json());
    }

    #[test]
    fn the_stream_is_a_session_progress_and_one_final_answer() {
        let adapter: Arc<dyn AgentAdapter> = Arc::new(Dsh);
        let mut p = StreamParser::new(adapter);

        assert!(p.line(r#"{"type":"session","sessionId":"session-50d3","cwd":"E:\\proj"}"#).is_empty());
        assert_eq!(p.session_id.as_deref(), Some("session-50d3"));
        assert!(p.line(r#"{"type":"status","phase":"turn_start","turn":1}"#).is_empty());
        assert!(p.line(r#"{"type":"status","phase":"step_start","turn":1,"step":1}"#).is_empty());

        let evs =
            p.line(r#"{"type":"status","phase":"turn_end","turn":1,"reason":{"kind":"error","error":{"message":"no API key","code":"MISSING_CREDENTIAL"}}}"#);
        match evs.as_slice() {
            [AgentEvent::Error { message, .. }] => assert_eq!(message, "DeepSeek Harness: no API key"),
            other => panic!("expected the turn's error, got {other:?}"),
        }

        assert!(p.line(r#"{"type":"final","text":""}"#).is_empty(), "an empty answer says nothing");
        let evs = p.line(r#"{"type":"final","text":"ok"}"#);
        match evs.as_slice() {
            [AgentEvent::TextDelta { text }] => assert_eq!(text, "ok"),
            other => panic!("expected the answer, got {other:?}"),
        }
        assert!(p.saw_result);

        // Events we haven't seen a shape for yet stay unheard, not mistaken for the answer.
        assert!(p.line(r#"{"type":"tool_call","name":"read"}"#).is_empty());
        assert_eq!(p.session_id.as_deref(), Some("session-50d3"), "an unheard event changes nothing");
    }
}
