use crux_core::{App, Command};

use crate::{
    effects::Effect,
    model::{Event, Model},
    view::ViewModel,
};

/// Crux 入口：连接事件、状态、Effect 与 ViewModel。
#[derive(Default)]
pub struct AppCore {}

impl App for AppCore {
    type Event = Event;
    type Model = Model;
    type ViewModel = ViewModel;
    type Effect = Effect;

    fn update(&self, event: Event, model: &mut Model) -> Command<Effect, Event> {
        model.update(event)
    }

    fn view(&self, model: &Model) -> ViewModel {
        model.into()
    }
}
