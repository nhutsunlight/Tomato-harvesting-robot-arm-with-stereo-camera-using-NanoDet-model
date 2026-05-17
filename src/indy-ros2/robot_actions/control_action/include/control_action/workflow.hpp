#pragma once

#include <memory>

#include <control_action/action_requests.hpp>
#include <control_action/control_checks.hpp>

#ifdef CONTROL_ACTION_USE_TEST_WORKFLOW
#include <control_action/test_execute.hpp>
#else
#include <control_action/execute_workflow.hpp>
#endif

namespace control_action
{

struct ExecuteWorkflow;
struct TestExecuteWorkflow;

struct Workflow
{
    template <typename Controller>
    static void execute(Controller& self,
                        const std::shared_ptr<typename Controller::GoalHandleControlRobot> goal_handle)
    {
#ifdef CONTROL_ACTION_USE_TEST_WORKFLOW
        TestExecuteWorkflow::execute(self, goal_handle);
#else
        ExecuteWorkflow::execute(self, goal_handle);
#endif
    }
};

}  // namespace control_action
