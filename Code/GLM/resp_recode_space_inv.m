function [responses_mat_noop_fire_right_left] = resp_recode_space_inv(n_frames_per_session, n_sessions, responses_mat)

responses_mat_noop_fire_right_left = false(n_frames_per_session, 4, n_sessions);

for sess_no = 1:n_sessions

    for frame_no = 1:n_frames_per_session

        resp_curr = responses_mat(frame_no,sess_no);

        switch resp_curr

            case 0

                responses_mat_noop_fire_right_left(frame_no,1,sess_no) = true;
                
            case 1

                responses_mat_noop_fire_right_left(frame_no,2,sess_no) = true;

            case 3

                responses_mat_noop_fire_right_left(frame_no,3,sess_no) = true;
                
            case 4

                responses_mat_noop_fire_right_left(frame_no,4,sess_no) = true;

            case 11

                responses_mat_noop_fire_right_left(frame_no,2,sess_no) = true;
                responses_mat_noop_fire_right_left(frame_no,3,sess_no) = true;
                %responses_mat_noop_fire_right_left(frame_no,5,sess_no) = true;

            case 12

                responses_mat_noop_fire_right_left(frame_no,2,sess_no) = true;
                responses_mat_noop_fire_right_left(frame_no,4,sess_no) = true;
                %responses_mat_noop_fire_right_left(frame_no,6,sess_no) = true;

        end

    end

end

responses_mat_noop_fire_right_left = double(responses_mat_noop_fire_right_left);

